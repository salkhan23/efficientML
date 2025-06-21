import torch
import transformers
import inspect
import tqdm
import torch.nn as nn
import datasets
import gc
from functools import partial

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
def pseudo_quantize_model_weight(model, n_bits, q_group_size):
    """
    Quantize all linear layers  based on the provided quantization group size and number of bits of quantization
    :param model:
    :param n_bits:
    :param q_group_size:
    :return:
    """

    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=n_bits, q_group_size=q_group_size)


def get_calibration_dataset(tokenizer=None, n_samples=256, block_size=512):
    """
    Loads a small calibration dataset from the 'pile-val-backup' validation split,
    filters and tokenizes short lines, and returns a list of fixed-length token blocks
    suitable for model calibration (e.g., for post-training quantization).

    The function:
    - Selects up to `n_samples` lines from the dataset that are shorter than `block_size` when tokenized.
    - Concatenates them into one long sequence.
    - Splits this sequence into multiple blocks of exactly `block_size` tokens.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to convert text to token IDs.
        n_samples (int): Number of valid text lines to collect before splitting into blocks.
        block_size (int): Desired number of tokens in each output block.

    Returns:
        List[torch.Tensor]: A list of tokenized input tensors, each of shape [1, block_size].

    Typical Use:
        This form of processing—concatenating multiple short samples and splitting them into
        fixed-size blocks—is widely used in LLM training and calibration. During training,
        models like GPT are fed sequences of a constant length (e.g., 512 or 1024 tokens) to
        enable efficient batching and hardware acceleration. For post-training quantization
        or calibration, fixed-size blocks ensure consistent activation statistics and allow
        for reproducible performance measurements. This strategy also minimizes padding and
        maximizes data utilization by packing short sequences together before splitting.

    Note:
        Lines longer than `block_size` tokens are skipped.
        Returned blocks are designed to be uniform-length and suitable for efficient batch inference
        or calibration tasks.
    """
    dataset_name = "mit-han-lab/pile-val-backup"
    print(f"Getting calibration dataset {dataset_name}")

    dataset = datasets.load_dataset(dataset_name, split="validation")
    dataset = dataset.shuffle(seed=42)

    # Debug  ----------------------------------------------------------
    # Dataset Info:
    # - Source: 'mit-han-lab/pile-val-backup' (validation split)
    # - Size: 214,670 samples
    # - Features:
    #     - 'text': raw natural language (variable-length strings)
    #     - 'meta': metadata with the original source
    #
    # Example:
    #     dataset['text'][0] → "Some natural language text..."
    #     dataset['meta'][0] → {'pile_set_name': 'Wikipedia (en)'}
    #
    # About The Pile:
    # - The Pile is an 825GB diverse, open-source text dataset created by EleutherAI.
    # - It combines 22 high-quality sources (e.g., Wikipedia, GitHub, ArXiv, PubMed)
    #   to support training and evaluating large language models (LLMs).
    #
    # About This Subset:
    # - 'pile-val-backup' is a small validation slice (~200K samples) created by MIT Han Lab,
    #   used for tasks like calibration, quantization, or small-scale evals.
    #
    # To load the full dataset:
    #     datasets.load_dataset("eleutherai/pile", split="train") (Requires significant disk space and memory.)
    # ----------------------------------------------------------

    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)  # [101, 102, 103, 104] dim=[4]

        if len(line_encoded) > block_size:
            continue

        sample = torch.tensor([line_encoded])
        # [line_encoded] → [[101, 102, 103, 104]]
        # torch.tensor([line_encoded] → [[101, 102, 103, 104]])
        # dim = [1,4]

        if sample.numel() == 0:
            continue

        samples.append(sample)

        n_run += 1

        if n_run == n_samples:
            break

    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")

    processed_tokenized_data = [
        cat_samples[:, i*block_size:(i+1)*block_size] for i in range(n_split)
    ]  # List length = n_split, each item shape = [1, block_size]

    print(f"Size of calibration Dataset: List of {len(processed_tokenized_data)}. Each element of shape "
          f"{processed_tokenized_data[0].shape}")

    return processed_tokenized_data


@torch.no_grad()
def get_calib_features(model, tokenizer):
    """
    Collects input activation statistics for all nn.Linear layers in the model
    using the calibration data.

    This is typically used for post-training quantization, where accurate input scale
    estimates (mean absolute values) are needed to determine quantization parameters
    like clipping thresholds or scale factors.

    The function:
    - Registers forward hooks on all nn.Linear layers in the model.
    - Runs the model on tokenized calibration samples.
    - For each linear layer, records the mean absolute value of its input features
      (per hidden dimension) for every sample.
    - Returns a dictionary mapping layer names to a list of recorded statistics.

    Args:
        model (torch.nn.Module): The model to collect activation stats from.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used to prepare calibration samples.

    Returns:
        dict[str, list[torch.Tensor]]: A dictionary mapping layer names to lists of
        1D tensors (mean absolute input activations per hidden unit).
    """
    input_dict = dict()  # Stores per-layer activation statistics

    # Hook function to collect stats on input activations.
    # Pytorch hooks have the format (module, input, output)
    # Here, functools.partial function is used to expand the hook to store the name of  module as well inside
    # input dict.
    def stat_input_max_hook(m, x, y, name1):
        # `m`: the layer (nn.Linear instance)
        # `x`: input to the layer, usually a tuple (we extract x[0],
        #      inputs inside the layers can have multiple elements, attention masks, head masks etc)
        # `y`: output from the layer
        # `name`: string identifier for the layer, passed via partial()

        if isinstance(x, tuple):
            x = x[0]

        # Compute per-channel (feature-wise) MEAN absolute activation
        x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        # x.shape = [batch_size, seq_len, hidden_dim]
        # x.view(-1, x.shape[-1]) FROM 3d TO 2d,  preserve the last dimension but allow the first dim to match
        # whatever size fits

        if name1 not in input_dict:
            input_dict[name1] = [x_max]
        else:
            input_dict[name1].append(x_max)

    hooks = []

    # Register forward hooks on all nn.Linear modules
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hook_fn = partial(stat_input_max_hook, name1=name)
            hooks.append(module.register_forward_hook(hook_fn))

    # Now parse the calibration dataset
    print("Collecting activation scales...")

    device = model.device
    samples = get_calibration_dataset(tokenizer)

    for input_ids in tqdm.tqdm(samples, desc="Calibrating"):
        input_ids = input_ids.to(device)
        model(input_ids)  # Triggers the hooks

    # Remove all hooks after we're done collecting
    for hook in hooks:
        hook.remove()

    return input_dict


@torch.no_grad()
def pseudo_quantize_model_salient_weight_fp16(model, n_bits, q_group_size, input_feat, preserve_ratio=0.01):
    """
    Applies pseudo quantization to the weights of all nn.Linear layers in the given model while preserving
    a small fraction of the most salient input channels in full precision.

    Salience is determined by the average absolute input activation per input channel, collected
    during a prior calibration run. The most important channels are excluded from quantization and
    their corresponding weights are restored to their original FP16 values after quantization.

    Args:
        model (torch.nn.Module): The model whose Linear layers are to be pseudo quantized.
        n_bits (int): Number of bits to use in quantization.
        q_group_size (int): Group size for quantization (e.g., 128 for group-wise quantization).
        input_feat (dict[str, list[Tensor]]): Dictionary mapping layer names to a list of per-channel
            input activation statistics (e.g., mean absolute activations).
        preserve_ratio (float, optional): Fraction of the most important input channels to preserve
            in full precision. Defaults to 0.01 (i.e., 1%).

    Returns:
        None. The function modifies the model weights in-place.
    """
    print("Starting the quantization process")

    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):

            if n not in input_feat:
                print(f"Layer {n} not found in input_feat. No activation statistics")
                continue  # Skip modules not in calibration stats

            # Aggregate importance from calibration stats
            importance = sum(input_feat[n]).float()

            # Step 1: Select most important channels based on input activation magnitude
            n_dim = importance.shape[0]
            n_preserve = max(int(n_dim * preserve_ratio), 1)
            _, outlier_indices = torch.topk(importance, n_preserve)
            assert outlier_indices.dim() == 1

            # Back up salient weight columns
            outlier = m.weight.data[:, outlier_indices].clone()

            # Quantize the entire weight tensor
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=n_bits, q_group_size=q_group_size)

            # Step 2: Restore the 1% (or preserve_ratio) most important channels
            for idx, outlier_idx in enumerate(outlier_indices):
                m.weight.data[:, outlier_idx].copy_(outlier[:, idx])


@torch.no_grad()
def pseudo_quantize_model_random_weight_fp16(model, n_bits, q_group_size, input_feat, preserve_ratio=0.01):
    """

    :param preserve_ratio:
    :param model:
    :param n_bits:
    :param q_group_size:
    :param input_feat:
    :return:
    """
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            importance = sum(input_feat[n]).float()
            n_dim = importance.shape[0]
            n_preserve = max(int(n_dim * preserve_ratio), 1)

            # Step 1: Randomly choose 1% of the weight channels
            outlier_mask = torch.randint(0, n_dim, (n_preserve,))
            assert outlier_mask.dim() == 1

            # Back up the values of the selected weight channels
            outlier = m.weight.data[:, outlier_mask].clone()

            # Quantize the entire weight tensor
            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=n_bits, q_group_size=q_group_size)

            # Restore the preserved weights
            for idx, outlier_idx in enumerate(outlier_mask):
                m.weight.data[:, outlier_idx].copy_(outlier[:, idx])


@torch.no_grad()
def pseudo_quantize_model_weight_scaleup(model, n_bit, q_group_size, input_feat, scale_factor, preserve_ratio=0.01):
    """
    Performs activation-aware quantization by scaling up important weight channels before quantization
    and scaling them back down afterward. This method helps protect salient weights from large
    quantization error while still using uniformly low bit-width quantization across all weights.

    Unlike mixed-precision approaches (e.g., GPTQ), which store important weights in higher precision
    (e.g., FP16) and quantize the rest, this technique preserves all weights in low bit-width (e.g., 4-bit)
    form, enabling more aggressive compression without sacrificing critical accuracy.

    The method:
    - Identifies the most important input channels to each Linear layer using their activation statistics.
    - Scales up these salient channels in the weight matrix by `scale_factor`.
    - Applies uniform symmetric quantization to all weights.
    - Scales down the salient channels post-quantization to preserve correct output behavior.

    This preserves model accuracy better at low bit-widths by reducing the quantization error of
    influential weights, without requiring changes to input activations or storing any weights in full precision.

    Args:
        model (torch.nn.Module): The model whose Linear layers will be quantized.
        n_bit (int): Number of bits to use for weight quantization.
        q_group_size (int): Number of weight channels per quantization group.
        input_feat (dict): Dictionary mapping Linear layer names to lists of input activation
                           magnitude tensors (used to determine importance).
        scale_factor (float): Multiplicative factor to apply to important weight channels before quantization.
        preserve_ratio (float, optional): Fraction of most important channels to scale/protect. Default is 0.01 (1%).

    Returns:
        None. The model is modified in-place.
    """
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            importance = sum(input_feat[n]).float()

            # Step 1: Select most important channels based on input activation magnitude
            n_dim = importance.shape[0]
            n_preserve = max(int(n_dim * preserve_ratio), 1)
            _, outlier_indices = torch.topk(importance, n_preserve)
            assert outlier_indices.dim() == 1

            # To simulate applying the scale factor, we can simply multiply it before quantization, and then
            # divide by the scale factor after quantization.
            # Scale up the values of the salient weight channels
            # m.weight.data[:, outlier_mask] *= scale_factor
            m.weight.data[:, outlier_indices] = m.weight.data[:, outlier_indices] * scale_factor

            # Quantize
            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=n_bit, q_group_size=q_group_size)

            # Step 2: Scale back down the values of the salient weight channels
            m.weight.data[:, outlier_indices] = m.weight.data[:, outlier_indices] / scale_factor


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

    # # debug visualize the model  ---------------
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

    # # Get model size  -----------------------------------------------------------------
    print(f"Model: {net._get_name()}")

    # Evaluate the model
    perplexity = evaluate(net, net_tokenizer)
    print("Full Model (No quantization)")
    print(f"Perplexity {perplexity:0.2f}")
    print(f"Size {get_model_size(net) / MiB:0.2f} MB")

    # Quantize the model ---------------------------------------------------------------

    # ----------------------------------------------------------------------------------
    # Naive Weight quantization based on weight magnitudes only
    # ----------------------------------------------------------------------------------
    print("\nQuantization based only on weight magnitudes ...")
    # test_tensor = torch.rand(128, 128)
    # pseudo_quantize_tensor(test_tensor, q_group_size=8)

    del net
    gc.collect()
    torch.cuda.empty_cache()
    net = transformers.AutoModelForCausalLM.from_pretrained(net_path, device_map="auto")

    w_bits = 3
    quantized_group_size = 128

    pseudo_quantize_model_weight(net, n_bits=w_bits, q_group_size=quantized_group_size)

    perplexity = evaluate(net, net_tokenizer)
    net_size = get_model_size(net, data_width_bits=w_bits, q_group_size=quantized_group_size)
    print(f"Perplexity {perplexity:0.2f}")
    print(f"Size {net_size / MiB:0.2f} MB")

    # ---------------------------------------------------------------------------------
    # Activation Aware Quantization - preserving top 1% outliers
    # ---------------------------------------------------------------------------------
    print("\nActivation Aware Quantization - preserving full precision of top outliers")

    del net
    gc.collect()
    torch.cuda.empty_cache()
    net = transformers.AutoModelForCausalLM.from_pretrained(net_path, device_map="auto")

    w_bits = 3
    quantized_group_size = 128

    input_features_stats = get_calib_features(net, net_tokenizer)
    pseudo_quantize_model_salient_weight_fp16(net, w_bits, quantized_group_size, input_features_stats)

    perplexity = evaluate(net, net_tokenizer)
    net_size = get_model_size(net, data_width_bits=w_bits, q_group_size=quantized_group_size)
    print(f"Perplexity {perplexity:0.2f}")
    print(f"Size {net_size / MiB:0.2f} MB")

    # ---------------------------------------------------------------------------------
    # Activation Aware Quantization - preserving 1% of random weights
    # ---------------------------------------------------------------------------------
    print("\nActivation Aware Quantization - preserving full precision of random outliers")

    del net
    gc.collect()
    torch.cuda.empty_cache()
    net = transformers.AutoModelForCausalLM.from_pretrained(net_path, device_map="auto")

    w_bits = 3
    quantized_group_size = 128

    input_features_stats = get_calib_features(net, net_tokenizer)
    pseudo_quantize_model_random_weight_fp16(net, w_bits, quantized_group_size, input_features_stats)

    perplexity = evaluate(net, net_tokenizer)
    net_size = get_model_size(net, data_width_bits=w_bits, q_group_size=quantized_group_size)
    print(f"Perplexity {perplexity:0.2f}")
    print(f"Size {net_size / MiB:0.2f} MB")

    # ----------------------------------------------------------------------------------------
    # Activation Aware Quantization - Scale up important weights, quantize and then scale down
    # ----------------------------------------------------------------------------------------
    print("\nActivation Aware Quantization - Scale Up, Quantize & scale down weights")

    del net
    gc.collect()
    torch.cuda.empty_cache()
    net = transformers.AutoModelForCausalLM.from_pretrained(net_path, device_map="auto")

    w_bits = 3
    quantized_group_size = 128

    input_features_stats = get_calib_features(net, net_tokenizer)
    pseudo_quantize_model_weight_scaleup(net, w_bits, quantized_group_size, input_features_stats, scale_factor=2)

    perplexity = evaluate(net, net_tokenizer)
    net_size = get_model_size(net, data_width_bits=w_bits, q_group_size=quantized_group_size)
    print(f"Perplexity {perplexity:0.2f}")
    print(f"Size {net_size / MiB:0.2f} MB")

    print("Effect of scaling factor in perplexity")
    scaling_factors_arr = [1, 2, 3, 4]
    perplexity_arr = []
    for scaling_factor in scaling_factors_arr:

        del net
        gc.collect()
        torch.cuda.empty_cache()
        net = transformers.AutoModelForCausalLM.from_pretrained(net_path, device_map="auto")
        
        pseudo_quantize_model_weight_scaleup(net, w_bits, quantized_group_size, input_features_stats, scaling_factor)
        perplexity = evaluate(net, net_tokenizer)
        perplexity_arr.append(perplexity)

    for sf_idx in range(len(perplexity_arr)):
        print(f"\t scaling factor {scaling_factors_arr[sf_idx]}: perplexity={perplexity_arr[sf_idx]:2f}")

    import pdb
    pdb.set_trace()
