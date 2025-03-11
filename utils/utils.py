import torch
import os
import dac

def save_model(model, optimizer, inf_context_length, filepath):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),

        'inf_context_length': inf_context_length,
        'embed_size': model.embed_size,
        'num_layers': model.num_layers, # len(model.layers),
        'num_heads': model.num_heads, #  model.layers[0].attention.num_heads,
        'forward_expansion': model.forward_expansion, # model.layers[0].feed_forward[0].out_features // model.embed_size,
        'dropout': model.dropout, # model.layers[0].dropout.p,
        'max_len': model.max_len, # model.position_embedding.num_embeddings,
        'num_codebooks': model.num_codebooks, 
        'vocab_size': model.vocab_size,
        'cond_size': model.cond_size,
        'use_adaLN' : model.use_adaLN,
        'num_classes': model.num_classes,
    }, filepath)

#-----------------------------------------------------

def load_model(filepath, TransformerClass, device='cuda', verbose=0):
    checkpoint = torch.load(filepath, map_location=device)  
    inf_context_length = checkpoint['inf_context_length'] # This is used to set the context length for the inference model
    
    model =  TransformerClass(
        embed_size=checkpoint['embed_size'],
        num_layers=checkpoint['num_layers'],
        num_heads=checkpoint['num_heads'],
        forward_expansion=checkpoint['forward_expansion'],
        dropout=checkpoint['dropout'],
        # This should be the training conext size size it affect the rotary positional encoding, not the conext length itself.
        max_len=checkpoint['max_len'],
        num_codebooks=checkpoint['num_codebooks'],
        vocab_size=checkpoint['vocab_size'],
        cond_size= checkpoint['cond_size'],
        use_adaLN= checkpoint['use_adaLN'],
        num_classes = checkpoint['num_classes'],
        verbose=verbose
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) #also restored lr

    model.to(device)

    # Move optimizer tensors to the correct device
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
            
    
    model.eval()
    return model, optimizer, inf_context_length, checkpoint['vocab_size'], checkpoint['num_codebooks'], checkpoint['cond_size'] # used for consructing the initial input window to the model
    
#----------------------------------------------------------------------    
# I´d like to get some better documentation on these DACFile params, but they are necessary for model.decompress(dacfile) to work
# 
def writeDACFile(fname, codeseq) :
    with torch.no_grad(): 
        dac_file = dac.DACFile(
                codes=codeseq.cpu(),
                chunk_length=codeseq.shape[2],
                original_length=int(codeseq.shape[2]*512), 
                input_db= torch.tensor(-20), #np.array([-20], dtype=np.float32),
                channels=1,
                sample_rate=44100,
                padding=True,
                dac_version='1.0.0'
            )
        
        # Save to disk
        directory = os.path.dirname(fname)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f'Just so ya know, I had to create the path to save the file')
        dac_file.save(fname + ".dac")
        

def generate_mask(sz, max_lookback):
    """
    Generates a mask for the attention mechanism that incorporates causal structure with a limited lookback window.

    Args:
    sz (int): Size of the square matrix (number of time steps/sequence length).
    max_lookback (int): Maximum number of steps a position can look back in the sequence.

    Returns:
    torch.Tensor: The attention mask.
    """
    # Full mask with all positions set to -inf initially
    mask = torch.full((sz, sz), float('-inf'))

    # Fill the band of allowed positions with 0s
    for i in range(sz):
        start = max(0, i - max_lookback)  # Start of the lookback window
        end = i + 1  # End of the lookback window, non-inclusive
        mask[i, start:end] = 0.0

    return mask



#############################################

def sample_top_n(logits, n):
    """
    Samples from the top-n highest probability logits for each token independently.

    Args:
        logits (torch.Tensor): Tensor of shape (batch_size, num_tokens, vocab_size).
        n (int): The number of top values to consider.

    Returns:
        torch.Tensor: Indices of sampled tokens, shape (batch_size, num_tokens).
    """
    batch_size, num_tokens, vocab_size = logits.shape

    # Get the top-n indices and values for each token independently
    top_n_values, top_n_indices = torch.topk(logits, n, dim=-1)  # Shape: (batch_size, num_tokens, n)

    # Convert top-n logits to probabilities using softmax
    top_n_probs = torch.nn.functional.softmax(top_n_values, dim=-1) # Shape: (ntokens, topn)

    # Sample from the top-n probabilities for each token independently
    sampled_idx = torch.multinomial(top_n_probs.view(batch_size * num_tokens, n), num_samples=1)  # Shape: (ntokens, 1)

    # Reshape back to (batch_size, num_tokens)
    sampled_idx = sampled_idx.view(batch_size, num_tokens)

    # Map sampled indices back to original vocabulary indices
    return torch.gather(top_n_indices, -1, sampled_idx.unsqueeze(-1)).squeeze(-1)  # Shape: (batch_size, num_tokens)

# # Example usage
# batch_size, num_tokens, vocab_size = 2, 5, 10
# logits = torch.randn(batch_size, num_tokens, vocab_size)  # Simulated logits
# n = 3
# sampled_tokens = sample_top_n(logits, n)
# print(sampled_tokens.shape)  # Expected shape: (batch_size, num_tokens)
# print(sampled_tokens)  # Indices sampled from the top-n choices for each token


########################################################################################################
# -----------------------------------------------------------------------
# The following are for creating sequences of parameters for inference
# -----------------------------------------------------------------------

def interpolate_vectors(v, s):
    assert len(v) == len(s), "List of vectors and list of time indexes must be of the same length."
    
    n = len(v[0])  # Length of each vector
    m = s[-1] + 1  # Last element of s plus one
    result = torch.zeros((1, m, n))  # Initialize the result tensor with zeros
    
    v_tensors = [torch.tensor(vec) for vec in v]
    
    for i in range(len(s) - 1):
        start_idx = s[i]
        end_idx = s[i + 1]
        start_vec = v_tensors[i]
        end_vec = v_tensors[i + 1]
        
        for j in range(start_idx, end_idx):
            t = (j - start_idx) / (end_idx - start_idx)
            result[0, j, :] = (1 - t) * start_vec + t * end_vec
    
    # Set the last vector directly
    result[0, s[-1], :] = v_tensors[-1]
    
    return result


# # Example usage
# v = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
# s = [0, 2, 4]
# result_tensor = interpolate_vectors(v, s)
# print(result_tensor)

# -----------------------------------------------------------------------
def breakpoints(allowed_keys, **kwargs):
    """
    Constructs a list of n tensors (rows) from keyword arguments.
    Meant to be interpreted as breakpoints in a time sequence
    
    Args:
        allowed_keys (list of str): List of keys that determine the column order.
        **kwargs: Each key must be one of allowed_keys and have a value that is a list.
                  All lists must be the same length, n.
                  
    Returns:
        list of torch.Tensor: A list of n tensors, each of shape [len(allowed_keys)].
        
    Example:
        allowed_keys = ["a", "b", "param"]
        result = create_tensor_rows(allowed_keys, a=[1, 2, 3], param=[4, 5, 6])
        # Expected rows:
        # Row 0: tensor([1, 0, 4])
        # Row 1: tensor([2, 0, 5])
        # Row 2: tensor([3, 0, 6])
    """
    # Determine n by checking the length of one of the value lists in kwargs.
    n = None
    for key, value_list in kwargs.items():
        if n is None:
            n = len(value_list)
        elif len(value_list) != n:
            raise ValueError("All value lists must have the same length.")
    if n is None:
        raise ValueError("No keyword arguments provided. At least one key must be specified.")

    # Build the list of tensors (each tensor is one row).
    tensor_rows = []
    for i in range(n):
        row_values = []
        for key in allowed_keys:
            # For keys not present in kwargs, default to 0.
            if key in kwargs:
                row_values.append(kwargs[key][i])
            else:
                row_values.append(0)
        tensor_rows.append(torch.tensor(row_values))
    return tensor_rows

# -----------------------------------------------------------------------
def timesegs(n):
    """
    Generate a list of time segment boundaries for partitioning the interval [0, 1]
    into n equal segments.

    For example, timesegs(4) returns:
        [0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1]

    Args:
        n (int): Number of equal segments.
    """    
    step=1/n
    stime=0
    tlist=[0]
    for i in range(1,n):
        tlist.extend([i*step,i*step])
    tlist.extend([1])
    return tlist
    
# -----------------------------------------------------------------------
def breakpoints_classseq(class_list, pvals, **kwargs):
    """
    Generates a conditioning sequence and corresponding time segments.
    The idea is that the resulting times-stamped vector sequence will 
    step through each class for an equal amount of time [normalized between [0,1])

    Args:
        class_list (list): A list of class names. 
        pvals (list): A list of values that will be appended to each tensor as its tail.
        **kwargs: Additional keyword arguments (currently not used).

    Returns:
        dict: A dictionary with two keys:
            'vsequence': A list of 2 * numclasses one-dimensional tensors. Each tensor has a length
                         of (numclasses + len(pvals)), where the first numclasses entries form a one-hot
                         vector and the last len(pvals) entries are the values from pvals.
            'vtimes': The time segments generated by timesegs(numclasses).

    Example:
        classes=['a','b','c']
        foo=conditioningTest(classes, [.5, .7])
        Returns:
        {'vsequence': [
            tensor([1.0000, 0.0000, 0.0000, 0.5000, 0.7000]),
            tensor([1.0000, 0.0000, 0.0000, 0.5000, 0.7000]),
            tensor([0.0000, 1.0000, 0.0000, 0.5000, 0.7000]),
            tensor([0.0000, 1.0000, 0.0000, 0.5000, 0.7000]),
            tensor([0.0000, 0.0000, 1.0000, 0.5000, 0.7000]),
            tensor([0.0000, 0.0000, 1.0000, 0.5000, 0.7000])],
         'vtimes': [0, 0.33, 0.33, 0.66, 0.66, 1]}

    """
    numclasses=len(class_list)
    vtimes=timesegs(numclasses)
    m=len(pvals)

    vsequence = []
    total_tensors = numclasses * 2
    vec_length = numclasses 


        # For each i from 0 to n-1, create two tensors.
    for i in range(numclasses):
        for j in range(2):
            t = torch.zeros(vec_length)
            # Set the "hot" position at index i.
            t[i] = 1.0
            # Set the last m elements to pval.
            t=torch.cat([t , torch.tensor(pvals)])
            vsequence.append(t)

    return { 'vsequence': vsequence, 'vtimes': vtimes}


def hotsequence(baselist, indexlist, val):
    """
    Creates a list of tensors for vsequence where each tensor is a copy of baselist
    except that it has a 1 at a specified index from indexlist.
    Each index in indexlist generates two identical tensors.

    Args:
        baselist (list or tensor): The base list/tensor to copy and modify.
        indexlist (list): List of indices where the value should be set to 1.

    Returns:
        list: A list of modified tensors (two per index).
    """
    # Convert baselist to a tensor if it isn't already
    base_tensor = torch.tensor(baselist, dtype=torch.float32)

    # List to store the modified tensors
    tensor_list = []

    for i in indexlist:
        modified_tensor = base_tensor.clone()  # Copy the base tensor
        modified_tensor[i] = val  # Set the specified index to val
        
        # Append two identical tensors
        tensor_list.append(modified_tensor.clone())
        tensor_list.append(modified_tensor.clone())

    vtimes=timesegs(len(indexlist))
    return { 'vsequence': tensor_list, 'vtimes': vtimes}


    