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

    print(f"Size of calibration Dataset: List of {len(processed_tokenized_data)} Values. Each element of shape "
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
        #      inputs inside the layers can have multiple elements, attention masks, head masks etc.)
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


@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    """
    Absorb activation-aware quantization scales into LayerNorm and Linear layers.

    This function implements an efficient form of activation-aware quantization by:
    - Pre-dividing LayerNorm weights and bias by the scales (effectively scaling down activations),
    - Compensating by multiplying the Linear layer weights by the same scales (scaling up weights),
    so that the usual quantize - dequantize steps are folded into existing network operations.

    This blending: Eliminates explicit scaling steps during inference,

    Args:
        ln (torch.nn.LayerNorm): LayerNorm module to rescale.
        fcs (torch.nn.Linear or list of torch.nn.Linear): One or more Linear layers sharing input features.
        scales (torch.Tensor): 1D tensor of scale factors per normalized feature.

    Why multiple Linear layers share the same scale:
    - They operate on the same normalized activations,
    - So a single scale vector suffices for consistent rescaling.
    """
    # Ensure fully connected layers (fcs) is a list for uniform processing
    if not isinstance(fcs, list):
        fcs = [fcs]

    # Move scales to the same device as LayerNorm parameters
    scales = scales.to(ln.weight.device)

    # Scale down LayerNorm (ln) weights and biases by dividing by scales
    ln.weight.div_(scales)  # in-place divide
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.div_(scales)

    # Scale up Linear weights by multiplying by scales to compensate
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))  # scales = [1, channel_in], weight = [channel_out, channel_in]

    # Sanity checks for NaNs after scaling
    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    """
    Efficiently folds activation-aware quantization scales into two connected Linear layers.

    PyTorch Linear weights have shape [out_features, in_features],
    with the first dimension as output channels.

    This function rescales:
    - The last N output channels of `fc1` (weights and bias),
    - The corresponding input channels of `fc2` (weights),

    where N = scales.size(0) is the number of channels being scaled.

    Notes on bias scaling:
    - The bias in `fc1` is scaled down along with its weights to keep its relative impact balanced;
      otherwise, the bias would become disproportionately large compared to scaled weights.
    - `fc2` weights are scaled up to compensate for the scaled-down outputs of `fc1`.
    - Biases in `fc2` are not scaled because scaling up the weights also scales the effect
      of the scaled-down bias from `fc1`. Since biases are added after multiplication,
      their relative effect remains correctly balanced without explicit scaling.

    Args:
        fc1 (torch.nn.Linear): First Linear layer producing activations.
        fc2 (torch.nn.Linear): Second Linear layer consuming those activations.
        scales (torch.Tensor): 1D tensor of scale factors for the last N output channels of `fc1`.

    Returns:
        None. Modifies the layers in-place.
    """

    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)

    scales = scales.to(fc1.weight.device)
    n_ch_2_scale = scales.size(0)

    # Scale down last N output channels of fc1 weights
    fc1.weight[-n_ch_2_scale:, :].div_(scales.view(-1, 1))  # scales reshaped to [N, 1] for broadcasting

    # Scale down corresponding bias terms of fc1 to keep balanced impact
    if fc1.bias is not None:
        fc1.bias[-n_ch_2_scale:].div_(scales)  # scales shape: [N]

    # Scale up last N input channels (columns) of fc2 weights
    # This compensates for the scaled-down output from fc1,
    # and also scales up the effect of the scaled-down bias implicitly.
    fc2.weight[:, -n_ch_2_scale:].mul_(scales.view(1, -1))  # [out_features, N] * [1, N]

    # Bias in fc2 is NOT scaled

    # Sanity checks to ensure no NaNs after scaling
    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def auto_scale_block(module, name, w_bit, q_group_size, input_feat):
    """
    Applies activation-aware scaling and quantization to a transformer block
    (specifically, an OPTDecoderLayer-like module).

    Steps:
    1. For self-attention's Q, K, V projections: compute shared input-based scale,
       fold into LayerNorm and Linear weights.
    2. For v_proj → out_proj: fold scale into this pair of Linear layers.
    3. For final LayerNorm → fc1: fold scale into LayerNorm and fc1.
    4. For fc1 → fc2: fold scale into this MLP connection.

    Args:
        module (nn.Module) : The transformer block to process (e.g., one encoder layer).
        name (str)         : Name prefix used to locate corresponding input activations in `input_feat`.
        w_bit (int)        : Number of bits to use for quantizing weights.
        q_group_size (int) : Number of output channels per quantization group.
        input_feat (dict[str, List[Tensor]]): Dictionary mapping layer names to lists of
           input activation tensors collected during a forward pass.

    Returns:
        None. The function modifies the module's weights in-place to embed scaling.
    :param module:
    :param name:
    :param w_bit:
    :param q_group_size:
    :param input_feat:
    :return:
    """
    def _search_module_scale(block, linears_to_scale, x, kwargs={}):
        """
        Find the exponent r* in [0,1] that minimizes the expected squared error:

        r* = argmin_{r} E[ || block(x) - block_quantized(x; scales = s_x^r) ||^2 ]

        where:
            - E[...] denotes the expected value over the input distribution,
            - block(x) is the original module output,
            - block_quantized(x; scales) is the output after scaling weights by scales, quantizing, and rescaling,
            - s_x is the given per-channel scale vector.

        Args:
            block (nn.Module): A module like self_attn or fc1/fc2 to forward with scaled weights.
            linears_to_scale (list of nn.Linear): Linear layers whose weights are scaled.
            x (Tensor)       : Calibration input features to feed into `block`.
            kwargs (dict)    : Optional forward kwargs for `block`.

        Returns:
            best_scales (Tensor): 1D tensor of scale factors minimizing MSE after quantization.
        """

        # block.parameters is generator that iterates through learnable parameters. Calling next once,
        # returns the first parameter.
        device = next(block.parameters()).device
        x = x.to(device)

        # Original (float32) output. If the block returns multiple outputs (e.g. attention weights),
        # keep only the first (output) values
        with torch.no_grad():
            original_output = block(x, **kwargs)
            if isinstance(original_output, tuple):
                original_output = original_output[0]

        # x.view(-1, x.shape[-1]) flattens batch and sequence dimensions  (keep only the embed dim)
        # .abs().mean(0) computes the avg activation magnitude per embedding dimension across all tokens
        mean_abs_activation = x.view(-1, x.shape[-1]).abs().mean(0)  # shape: [embed_dim]

        best_error = float("inf")
        best_scales = mean_abs_activation  # starting scales
        # Save a CPU copy of the block’s org. weights before scaling to preserve them.
        # After each scaling iteration, restore these originals to avoid cumulative modifications.
        original_state = {k: v.cpu() for k, v in block.state_dict().items()}

        n_grid = 20  # Number of exponents to try for s = |x|^r

        # print(f"Finding optimization scales for {block}")

        for i in range(n_grid):
            ratio = i / n_grid
            scales = mean_abs_activation.pow(ratio)

            # Clamp min scale to a small value so that divide by the norm does not give a NaN
            epsilon = 1e-8
            scales = torch.clamp(scales, min=epsilon)

            # Normalize and clip
            # Normalization prevents the scales from becoming arbitrarily large or small, which could
            # distort weight values and make quantization error comparisons invalid. It ensures the
            # optimization focuses on the relative scaling between channels.  Normalization by the
            # geometric mean balances the scales multiplicatively, centering them between their min and
            # max values to handle wide dynamic ranges effectively. However, normalization does not remove
            # extreme outliers, so clipping is applied to restrict scales
            norm = torch.sqrt(scales.max() * scales.min())
            scales = torch.clamp(scales / norm, min=1e-4, max=1e4)

            # print(f"{i}: Ratio {ratio}, scale max {scales.max()}, min {scales.min()}. "
            #       f"Any NaNs {torch.isnan(scales).any()}")

            for fc in linears_to_scale:
                scales = scales.to(fc.weight.device)

                # Temporarily scale weight, quantize, then rescale back
                # We are just interested in the scaling factor at the moment and this
                # approach models the impact of including them. But it adds extra steps to the process
                # later scales will be added more efficiently in an optimized way (scale_ln_fcs, scale_fc_fc)
                fc.weight.mul_(scales)
                fc.weight.data = pseudo_quantize_tensor(fc.weight.data, w_bit, q_group_size)
                fc.weight.div_(scales)

            # Forward with quantized weights
            q_output = block(x, **kwargs)
            if isinstance(q_output, tuple):
                q_output = q_output[0]

            # Measure quantization error
            error = (original_output - q_output).float().pow(2).mean().item()
            if error < best_error:
                best_error = error
                best_scales = scales.clone()

            # Restore original state for next trial
            block.load_state_dict(original_state)

        assert torch.isnan(best_scales).sum() == 0, "NaNs in best scales"
        return best_scales.detach()

    # attention input
    # ------------------------------------------------------------------------------------
    # 1. Scale Q, K, V projections using common input from self_attn_layer_norm
    #
    # * Use the same scale for Q, K, and V? Q, K, and V are projections of the same input tensor,
    # so a single per-channel scale vector can be shared across them.
    #
    # * The activations from the OUTPUT of the self attention block are used.  It’s used as a proxy
    # because the true input to the Q/K/V projections wasn't explicitly saved during calibration.
    # Ideally, scaling should be computed from the actual input to the Q/K/V layers for correctness.
    # ------------------------------------------------------------------------------------
    # print(f"Starting quantization of self attention layer Q,K,V weight matrices for {name} ")
    qkv_input = input_feat[name + '.self_attn.out_proj']
    qkv_input = torch.cat([x.unsqueeze(0) for x in qkv_input], dim=0).unsqueeze(0)

    qkv_linears = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
    qkv_scales = _search_module_scale(module.self_attn, qkv_linears, qkv_input)
    # add optimal scales efficiently - Assume a pre-norm arch, layer norm, then projection layers
    scale_ln_fcs(module.self_attn_layer_norm, qkv_linears, qkv_scales)

    # ------------------------------------------------------------------------------------
    # 2. Scale attention output projection (v_proj → out_proj)
    # ------------------------------------------------------------------------------------
    # print(f"Starting quantization of self attention output weight matrices for {name} ")
    out_proj_input = input_feat[name + '.self_attn.out_proj']
    out_proj_input = torch.cat([x.unsqueeze(0) for x in out_proj_input], dim=0)

    out_proj_scales = _search_module_scale(
        module.self_attn.out_proj, [module.self_attn.out_proj], out_proj_input)
    # add output scales efficiently, div v_projection layer weights, multiply output projection layer weights
    scale_fc_fc(module.self_attn.v_proj, module.self_attn.out_proj, out_proj_scales)

    # ------------------------------------------------------------------------------------
    # Step 3: Quantize MLP first layer (fc1), coming from final_layer_norm
    # ------------------------------------------------------------------------------------
    # print(f"Starting quantization of MLP FC1 layer for {name} ")
    fc1_input = input_feat[name + '.fc1']
    fc1_input = torch.cat([x.unsqueeze(0) for x in fc1_input], dim=0)
    fc1_scales = _search_module_scale(module.fc1, [module.fc1], fc1_input)
    # again assuming pre-norm config
    scale_ln_fcs(module.final_layer_norm, module.fc1, fc1_scales)

    # ------------------------------------------------------------------------------------
    # Step 4: Quantize MLP second layer (fc2), which takes input from fc1
    # ------------------------------------------------------------------------------------
    # print(f"Starting quantization of MLP FC2 layer for {name} ")
    fc2_input = input_feat[name + '.fc2']
    fc2_input = torch.cat([x.unsqueeze(0) for x in fc2_input], dim=0)
    fc2_scales = _search_module_scale(module.fc2, [module.fc2], fc2_input)
    # notice the cascading on fc1. scaled up for its own optimal scale,
    # then scaled down for the div of fc2 scales
    scale_fc_fc(module.fc1, module.fc2, fc2_scales)


@torch.no_grad()
def pseudo_quantize_model_weight_auto_scale(
    model, w_bit, q_group_size, input_feat
):
    from transformers.models.opt.modeling_opt import OPTDecoderLayer

    # Scale the weights optimally
    for name, module in model.named_modules():
        print(f"{name} is OPTDECODERLayer {isinstance(module, OPTDecoderLayer)}")
        if isinstance(module, OPTDecoderLayer):
            auto_scale_block(module, name, w_bit, q_group_size, input_feat)

    # Actually do the quantization
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

    # # Evaluate the model
    # perplexity = evaluate(net, net_tokenizer)
    # print("Full Model (No quantization)")
    # print(f"Perplexity {perplexity:0.2f}")
    # print(f"Size {get_model_size(net) / MiB:0.2f} MB")
    #
    # # Quantize the model ---------------------------------------------------------------
    #
    # # ----------------------------------------------------------------------------------
    # # Naive Weight quantization based on weight magnitudes only
    # # ----------------------------------------------------------------------------------
    # print("\nQuantization based only on weight magnitudes ...")
    # # test_tensor = torch.rand(128, 128)
    # # pseudo_quantize_tensor(test_tensor, q_group_size=8)
    #
    # del net
    # gc.collect()
    # torch.cuda.empty_cache()
    # net = transformers.AutoModelForCausalLM.from_pretrained(net_path, device_map="auto")
    #
    # w_bits = 3
    # quantized_group_size = 128
    #
    # pseudo_quantize_model_weight(net, n_bits=w_bits, q_group_size=quantized_group_size)
    #
    # perplexity = evaluate(net, net_tokenizer)
    # net_size = get_model_size(net, data_width_bits=w_bits, q_group_size=quantized_group_size)
    # print(f"Perplexity {perplexity:0.2f}")
    # print(f"Size {net_size / MiB:0.2f} MB")
    #
    # # ---------------------------------------------------------------------------------
    # # Activation Aware Quantization - preserving top 1% outliers
    # # ---------------------------------------------------------------------------------
    # print("\nActivation Aware Quantization - preserving full precision of top outliers")
    #
    # del net
    # gc.collect()
    # torch.cuda.empty_cache()
    # net = transformers.AutoModelForCausalLM.from_pretrained(net_path, device_map="auto")
    #
    # w_bits = 3
    # quantized_group_size = 128
    #
    # input_features_stats = get_calib_features(net, net_tokenizer)
    # pseudo_quantize_model_salient_weight_fp16(net, w_bits, quantized_group_size, input_features_stats)
    #
    # perplexity = evaluate(net, net_tokenizer)
    # net_size = get_model_size(net, data_width_bits=w_bits, q_group_size=quantized_group_size)
    # print(f"Perplexity {perplexity:0.2f}")
    # print(f"Size {net_size / MiB:0.2f} MB")
    #
    # # ---------------------------------------------------------------------------------
    # # Activation Aware Quantization - preserving 1% of random weights
    # # ---------------------------------------------------------------------------------
    # print("\nActivation Aware Quantization - preserving full precision of random outliers")
    #
    # del net
    # gc.collect()
    # torch.cuda.empty_cache()
    # net = transformers.AutoModelForCausalLM.from_pretrained(net_path, device_map="auto")
    #
    # w_bits = 3
    # quantized_group_size = 128
    #
    # input_features_stats = get_calib_features(net, net_tokenizer)
    # pseudo_quantize_model_random_weight_fp16(net, w_bits, quantized_group_size, input_features_stats)
    #
    # perplexity = evaluate(net, net_tokenizer)
    # net_size = get_model_size(net, data_width_bits=w_bits, q_group_size=quantized_group_size)
    # print(f"Perplexity {perplexity:0.2f}")
    # print(f"Size {net_size / MiB:0.2f} MB")
    #
    # # ----------------------------------------------------------------------------------------
    # # Activation Aware Quantization - Scale up important weights, quantize and then scale down
    # # ----------------------------------------------------------------------------------------
    # print("\nActivation Aware Quantization - Scale Up, Quantize & scale down weights")
    #
    # del net
    # gc.collect()
    # torch.cuda.empty_cache()
    # net = transformers.AutoModelForCausalLM.from_pretrained(net_path, device_map="auto")
    #
    # w_bits = 3
    # quantized_group_size = 128
    #
    # input_features_stats = get_calib_features(net, net_tokenizer)
    # pseudo_quantize_model_weight_scaleup(net, w_bits, quantized_group_size, input_features_stats, scale_factor=2)
    #
    # perplexity = evaluate(net, net_tokenizer)
    # net_size = get_model_size(net, data_width_bits=w_bits, q_group_size=quantized_group_size)
    # print(f"Perplexity {perplexity:0.2f}")
    # print(f"Size {net_size / MiB:0.2f} MB")
    #
    # print("Effect of scaling factor in perplexity")
    # scaling_factors_arr = [1, 2, 3, 4]
    # perplexity_arr = []
    # for scaling_factor in scaling_factors_arr:
    #
    #     del net
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     net = transformers.AutoModelForCausalLM.from_pretrained(net_path, device_map="auto")
    #
    #     pseudo_quantize_model_weight_scaleup(net, w_bits, quantized_group_size, input_features_stats, scaling_factor)
    #     perplexity = evaluate(net, net_tokenizer)
    #     perplexity_arr.append(perplexity)
    #
    # for sf_idx in range(len(perplexity_arr)):
    #     print(f"\t scaling factor {scaling_factors_arr[sf_idx]}: perplexity={perplexity_arr[sf_idx]:2f}")

    # -------------------------------------------------------------------------------------
    # Finding the optimal scaling factor
    # -------------------------------------------------------------------------------------
    print("\nActivation Aware Quantization - Automatically finding the scale factor for activation aware training")

    del net
    gc.collect()
    torch.cuda.empty_cache()
    net = transformers.AutoModelForCausalLM.from_pretrained(net_path, device_map="auto")

    input_features_stats = get_calib_features(net, net_tokenizer)
    pseudo_quantize_model_weight_auto_scale(net, w_bit=3, q_group_size=128, input_feat=input_features_stats)

    # Evaluate the model
    model_perplexity = evaluate(net, net_tokenizer)
    model_size = get_model_size(net, data_width_bits=3, q_group_size=128)
    print(f"\nmodel perplexity: {model_perplexity:.2f}")
    print(f"model size: {model_size / MiB:.2f} MiB")
    import pdb
    pdb.set_trace()
