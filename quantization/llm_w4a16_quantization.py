import torch
import transformers
import inspect
import tqdm
import torch.nn as nn
import datasets
import gc

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB


def get_model_size(model, data_width_bits=16, q_group_size=-1):
    """
    Get Model Size in bits.

    Counts the # elements of all parameters of a model, multiply with daw_width_size
    Can also handle models quantizers with group quantization. per group 16 bit fp for scale and 4 bit zero
    crossing consumption is assumed.

    :param model:
    :param data_width_bits:
    :param q_group_size:
    :return:
    """
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()

    # Include per-bit contribution for using a scale and zero point
    if q_group_size != -1:
        data_width_bits += (16 + 4) / q_group_size

    size_bits = total_params * data_width_bits

    return size_bits


# core quantization method (simulated quantization)
def pseudo_quantize_tensor(w, n_bit=4, q_group_size=-1):
    """
    simulated quantization of a weight(must be 2D)

    Uniform Quantization: Uniform quantization maps real-valued inputs in the range [β, α] to integer
    values in the range [0, 2^b - 1], where b is the number of quantization bits.

    Notation:
        Quantized weight: w_q
        Scale factor: s_q
        Zero point: zp

    (1) scale             = s_q = (α - β) / (2^b - 1)
    (2) Zero point        = zp = - Round(β / s_q)
    (3) quantized weights = w_q = Clamp(Round(w / s_q) + z)

    :param w:
    :param n_bit:
    :param q_group_size:
    :return:
    """
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0  # ensure the last dimension is divisible by
        # valid group quantization value

    w = w.reshape(-1, q_group_size)  # reshape to [org_w_shape[0] * org_w_shape[1]/q_group_size, q_group_size]
    assert w.dim() == 2

    min_w, _ = w.min(dim=-1)

    # now do groupwise quantization
    # -----------------------------------------------------------------------------------
    # Calculate the maximum (\alpha) and minimum values (\beta) in the tensor.
    max_val = w.amax(dim=1, keepdim=True)
    assert max_val.dim() == 2 and max_val.size(0) == w.size(0) and max_val.size(1) == 1

    min_val = w.amin(dim=1, keepdim=True)
    assert min_val.dim() == 2 and min_val.size(0) == w.size(0) and min_val.size(1) == 1

    # scale and zero pint calculations
    max_int = 2 ** n_bit - 1

    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    assert scales.shape == max_val.shape

    zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)
    assert scales.shape == min_val.shape

    # Check for any Nan Values
    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    # Quantize W: Map values in the range [\beta, \alpha] to lie within [0, 2^b - 1] (Formula 3)
    w = torch.clamp(torch.round(w / scales) + zeros, 0, max_int)
    assert w.dim() == 2 and w.size(0) == scales.size(0) and w.size(1) == q_group_size

    # De-quantize W (pseudo quantization, the inverse transformation of Formula 3)
    w = (w - zeros) * scales
    assert w.dim() == 2 and w.size(0) == scales.size(0) and w.size(1) == q_group_size

    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)
    return w


@torch.no_grad()
def pseudo_quantize_model_weight(model, w_bit, q_group_size):

    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, q_group_size=q_group_size)


def evaluate(model, tokenizer):
    """
    Evaluate a model over the wikitext (version: wikitext-2-raw-v1) test set and calculate its perplexity
    (average negative log likelihood)

    :param model:
    :param tokenizer:
    :return:
    """
    # ---------------------------------------------------------------------------------------------
    # Load the data & tokenize it
    # ---------------------------------------------------------------------------------------------
    # Loads wikitext dataset version from GitHub ( https://huggingface.co/datasets/Salesforce/wikitext)
    test_enc = datasets.load_dataset(
        path='wikitext',
        name='wikitext-2-raw-v1',
        split='test')

    # # Print available version of the dataset
    # configs = datasets.get_dataset_config_names("wikitext")
    # print(f" Available versions of wikitext dataset configs:\b {configs}")

    # # # Print examples of the dataset
    # print(f"Dataset Examples:")
    # count = 10
    # for c_idx in range(count):
    #     print(f"{c_idx:} {test_enc[c_idx]['text'][:200]}")

    test_enc = tokenizer("\n\n".join(test_enc['text']), return_tensors='pt')
    # [1] Join all articles into one long string (separated by "\n\n").
    # [2] Tokenize the full string into PyTorch-style tensors.

    # Loading as a single string, is commonly practice in causal LLM training (predict next token)
    # This avoids the need to handle document boundaries as all data is treated as one stream.

    # The entire data is loaded into the memory. Tokenizers run on the CPU and make batch size data
    # available to the GPU at run time. On the CPU there is ample memory (RAM size) for small to medium
    # datasets. For large data sets or environments with memory constraints, use chunk processing with custom
    # dataLoader to handle chunk transitions. Hugging Face also has a streaming mode that can be used.

    # OUTPUT
    # test_enc is a dict with:
    #     input_ids: token IDs tensor, shape (1, sequence_length)
    #     attention_mask: all ones tensor (no padding. Useful when developing the data loader)

    # # DEBUG
    # print(f"Number of Tokens in Dataset {test_enc['input_ids'].shape[1]}")
    #
    # pos = 10
    # token_id = test_enc['input_ids'][0, pos].item()  # get token ID at batch 0, position 10
    # token_str = tokenizer.convert_ids_to_tokens(token_id)  # convert token ID to token string
    # print(f"Example token at position {pos}: ID={token_id}, token='{token_str}'")

    # --------------------------------------------------------------------------------------------
    # Get Model predictions
    # --------------------------------------------------------------------------------------------
    test_enc = test_enc.input_ids.to(model.device)  # test_enc is now only the message ids

    n_batches = 40
    seq_len = 2048
    model = model.eval()
    total_tokens = 0  # total tokens counted

    loss_fcn = nn.CrossEntropyLoss()
    nlls = []  # negative log likelihood for perplexity calculation

    for b_idx in tqdm.tqdm(range(n_batches), desc="evaluating..."):

        batch = test_enc[:,  b_idx * seq_len: (b_idx+1) * seq_len]  # dim = [1, len(data)]

        # Ensures the batch is at least two tokens long, which is the minimum
        # needed to compute a next-token prediction.
        if batch.shape[1] < 2:
            continue

        inputs = batch.to(model.device)

        with torch.no_grad():
            output = model(inputs)

        logits = output.logits  # [1m, batch_size, embed_dim]

        # ---------------------------------------------------------------------------------
        # calculate loss
        # ----------------------------------------------------------------------------------
        # Causal Language Model predict next token.
        # There is no prediction for first token (needs to be removed from the labels) and
        # there is no corresponding label for the last model output.
        # TO align the two, both the labels and the logits need to be shifted (but differently)

        # shift the logits and labels to align
        # Last logit needs to be ignored. No label for it
        shifted_logits = logits[:, :-1, :]  # ignore the last output  [1, batch_size-1, embed_dim]
        shifted_labels = inputs[:, 1:]      # ignore the first label  [1, batch_size-1, embed_dim]

        # reshape the outputs to be in a format expected by cross entropy
        embed_dim = shifted_logits.shape[-1]
        shifted_logits = shifted_logits.reshape(1 * (seq_len - 1), embed_dim)  # [seq_len, embed_dim]
        shifted_labels = shifted_labels.reshape(1 * (seq_len - 1))  # [seq_len]

        loss = loss_fcn(shifted_logits, shifted_labels)

        n_batch_tokens = shifted_labels.numel()
        neg_log_likelihood = loss.item() * n_batch_tokens
        nlls.append(neg_log_likelihood)
        total_tokens += n_batch_tokens

    avg_neg_log_likelihood = sum(nlls) / total_tokens  # average loss per token over all batches

    return torch.exp(torch.tensor(avg_neg_log_likelihood))  # perplexity


if __name__ == "__main__":
    net_path = "facebook/opt-1.3b"
    # net_path = 'distilgpt2'

    # Get a Model & its corresponding tokenizer ---------------------------------------------------

    net = transformers.AutoModelForCausalLM.from_pretrained(net_path, device_map="auto")
    # device_map = "auto", tells transformer library how to deply the model on GPU/CPU, so it
    # fits & runs efficiently.

    net_tokenizer = transformers.AutoTokenizer.from_pretrained(net_path, use_fast=False)

    # # debug visualize the model  -----------------------------------------------------------------
    # print(f"Model overall Structure: {'-'*50}")
    # print(net)
    # print(f"{'-' * 80}")
    #
    # print(f"Full Structure of model")
    # for name, module in net.named_modules():
    #     print(name)
    # print(f"{'-' * 80}")
    #
    # print(f"Individual Code layer")
    # print(inspect.getsource(net.model.decoder.layers[0].forward))
    # # print(inspect.getsource(net.transformer.h[0].forward))

    # Get model size  -----------------------------------------------------------------
    net_size_bits = get_model_size(net)
    print(f"Model: {net._get_name()}")
    print(f"Model Size {net_size_bits/MiB:0.2f} MB")

    # # evaluate the model
    perplexity = evaluate(net, net_tokenizer)
    print(f"Model perplexity {perplexity:0.2f}")

    # Quantize the model ---------------------------------------------------------------
    # test_tensor = torch.rand(128, 128)
    # pseudo_quantize_tensor(test_tensor, q_group_size=8)

    del net
    gc.collect()
    torch.cuda.empty_cache()

    net = transformers.AutoModelForCausalLM.from_pretrained(net_path, device_map="auto")
    pseudo_quantize_model_weight(net, w_bit=3, q_group_size=128)

    # Evaluate the model
    model_perplexity = evaluate(net, net_tokenizer)
    model_size = get_model_size(net, data_width_bits=3, q_group_size=128)
    print(f"\nQuantized Model")
    print(f"\nmodel perplexity: {model_perplexity:.2f}")
    print(f"quantized model size: {model_size / MiB:.2f} MiB")

    import pdb
    pdb.set_trace()
