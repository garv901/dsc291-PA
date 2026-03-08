import argparse
import json

def model_training_cost_analysis(model_config):
    # TODO: added your code here
    with open(model_config, "r") as f:
        model_config = json.load(f)
    # You are free to add any helper functions and import any packages you see fit in this file
    V = model_config["vocab_size"]
    S = model_config["max_sequence_length"]
    D = model_config["hidden_size"]
    F = model_config["intermediate_size"]
    P = model_config["max_position_embeddings"]
    N = model_config["num_attention_heads"]
    L = model_config["num_hidden_layers"]
    H = D // N
    # num_parameters = L * D * (3 * F + 2 * H * (N + N) + 2) + D * V + P * D + D

    word_embedding_params = V * D
    positional_embedding_params = P * D

    mlp_params = 3 * D * F
    attention_params = 4 * D * D # 2 * H * (N + N) * D
    norm_params = 2 * D

    per_layer_params = attention_params + mlp_params + norm_params
    num_parameters = word_embedding_params + positional_embedding_params + L * per_layer_params
    num_flops = 2 * S * (attention_params + mlp_params)

    layer_param_memory = per_layer_params * 2
    activation_memory = (
        S * D +          # hidden states
        3 * S * D +      # q, k, v
        S * F            # mlp intermediate
    ) * 2

    memory_cost = (layer_param_memory + activation_memory) / (1024 ** 3)

    return num_parameters, num_flops / 1e12, memory_cost

def get_optimal_N_D(training_budget):
    # TODO: added your code here
    # You are free to add any helper functions and import any packages you see fit in this file
    raise NotImplementedError()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training cost analysis')
    parser.add_argument('--model_config', type=str, help='Path to model config file')
    parser.add_argument('--training_budget', type=float, default=None, help='Training budget')
    args = parser.parse_args()

    if args.model_config:
        num_parameters, num_flops, memory_cost = model_training_cost_analysis(args.model_config)
        print(f"Number of parameters: {num_parameters}")
        print(f"Number of TFLOPs: {num_flops}")
        print(f"Peak memory cost: {memory_cost} GBs")

    if args.training_budget:    
        N, D = get_optimal_N_D(args.training_budget)
        print(f"Optimal N: {N}")
        print(f"Optimal D: {D}")

    