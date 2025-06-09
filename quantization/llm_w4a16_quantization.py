"""------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------- """
import tqdm
import torch
from torch import nn
import transformers
import datasets
from functools import partial
import gc  # garbage collection


def evaluate(model, tokenizer):
    """

    :param model:
    :param tokenizer:
    :return:
    """
    test_enc = datasets.load_dataset(
        path='wikitext',
        name='wikitext-2-raw-v1',
        split='test')
    # Load wikitext dataset version from GitHub ( https://huggingface.co/datasets/Salesforce/wikitext)
    # The WikiText language modeling dataset is a collection of over 100 million tokens extracted
    # from the set of verified Good and Featured articles on Wikipedia.

    # # Print available version of the dataset
    # configs = datasets.get_dataset_config_names("wikitext")
    # print(f" Available versions of wikitext dataset configs:\b {configs}")

    # # Print examples of the dataset
    print(f"Dataset Examples:")
    count = 10
    for c_idx in range(count):
        print(f"{c_idx:} {test_enc[c_idx]['text'][:200]}")

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

    test_enc = test_enc.input_ids.to(model.device)
    # test_enc is now only the message ids

    n_batches = 40
    batch_size = 1024
    model = model.eval()
    total_tokens = 0  # total tokens counted

    nlls = []  # negative log likelihood.

    for i in tqdm.tqdm(range(n_batches), desc="evaluating..."):

        batch = test_enc[:, (i * batch_size):((i + 1) * batch_size)].to(model.device)

        with torch.no_grad():
            output = model(batch)  # ordered dictionary with keys (logits, past_value keys)

        lm_logits = output.logits
        # lm_logits dim = [batch, seq_len, embed_dim]

        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        # contiguous aligns all memory into a contiguous block
        # shift_logits dim = [batch, seq_len - 1, embed_dim] = [1, 1023, emb_dim]

        shift_labels = test_enc[:, (i * batch_size): ((i + 1) * batch_size)]  # dim = [1,1024]
        shift_labels = shift_labels[:, 1:]  # dim = [1,1023]

        # get rid of the batch dimension
        shift_labels = shift_labels.view(-1)
        shift_logits = torch.squeeze(shift_logits)

        loss_fcn = nn.CrossEntropyLoss()
        loss = loss_fcn(shift_logits, shift_labels)  # loss_fcn([1023, vocab_size], [1023]) = scalar [mean over batch]

        # Calculate Perplexity.
        # Perplexity = exp(average negative log likelihood per token)

        n_batch_tokens = shift_labels.numel()  # number of tokens in this batch (batch_size -1)
        neg_log_likelihood = loss.item() * n_batch_tokens  # total loss for this batch (mean * count)

        nlls.append(neg_log_likelihood)
        total_tokens += n_batch_tokens

    avg_neg_log_likelihood = sum(nlls) / total_tokens  # average loss per token over all batches

    return avg_neg_log_likelihood


if __name__ == "__main__":
    # Usage example:
    net_name = "distilgpt2"  # Lighter version of GPT-2 (~82M params)

    net_tokenizer = transformers.AutoTokenizer.from_pretrained(net_name)
    net = transformers.AutoModelForCausalLM.from_pretrained(net_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    net = net.to(device)

    evaluate(net, net_tokenizer)

    import pdb
    pdb.set_trace()
