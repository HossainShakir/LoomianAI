import torch
import pickle
from torch import nn

# First, define your model class (must match original architecture)
class LoomianPredictor(nn.Module):
    def __init__(self, vocab_size, rank_size, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rank_embedding = nn.Embedding(rank_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim * 2, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input_ids, rank):
        loomian_emb = self.embedding(input_ids)
        rank_emb = self.rank_embedding(rank).unsqueeze(1).expand(-1, input_ids.size(1), -1)
        combined = torch.cat([loomian_emb, rank_emb], dim=-1)
        lstm_out, _ = self.lstm(combined)
        logits = self.fc(lstm_out)
        return logits

def load_loomian_model(weights_path='loomian_model_weights.pth', 
                      assets_path='loomian_model_assets.pkl'):
    with open(assets_path, 'rb') as f:
        assets = pickle.load(f)
    
    model_config = {
        'vocab_size': assets['model_config']['vocab_size'],
        'rank_size': assets['model_config']['rank_size'],
        'embedding_dim': assets['model_config']['embedding_dim'],
        'hidden_dim': assets['model_config']['hidden_dim']
    }
    
    model = LoomianPredictor(**model_config)
    model.load_state_dict(torch.load(weights_path, weights_only=True))  
    model.eval()
    
    return model, assets['tokenizer'], assets['rank_encoder'], assets['max_len']

model, tokenizer, rank_encoder, max_len = load_loomian_model()

reverse_tokenizer = {v: k for k, v in tokenizer.items()}
special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

def predict_team(partial_team, rank, model, tokenizer, rank_encoder, max_length=7, max_attempts=100):
    model.eval()
    
    input_tokens = [tokenizer.get(l, tokenizer['[UNK]']) for l in partial_team]
    input_tensor = torch.tensor([input_tokens + [tokenizer['[PAD]']] * (max_len - len(input_tokens))])
    rank_tensor = torch.tensor([rank_encoder[rank]])
    
    predicted_team = partial_team.copy()
    attempts = 0
    
    while len(predicted_team) < max_length and attempts < max_attempts:
        attempts += 1
        with torch.no_grad():
            logits = model(input_tensor, rank_tensor)
        
        top_tokens = torch.topk(logits[0, len(predicted_team)-1], 5).indices.tolist()
        
        for token_id in top_tokens:
            next_token = reverse_tokenizer[token_id]
            if (next_token not in predicted_team and 
                next_token not in special_tokens and
                next_token != '[PAD]'):
                predicted_team.append(next_token)
                input_tokens = [tokenizer.get(l, tokenizer['[UNK]']) for l in predicted_team]
                input_tensor = torch.tensor([input_tokens + [tokenizer['[PAD]']] * (max_len - len(input_tokens))])
                break
    
    return predicted_team[:max_length]

def interactive_predictor_case_insensitive(model, tokenizer, rank_encoder):
    print("=== Loomian Team Predictor ===")
    print("Enter your partial team and rank to get predictions")
    print("Type 'quit' at any time to exit\n")
    
    loomian_lower_map = {k.lower(): k for k in tokenizer.keys() if k not in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']}
    rank_lower_map = {k.lower(): k for k in rank_encoder.keys()}
    
    while True:
        try:
            team_input = input("Enter partial team (comma separated Loomians): ").strip()
            if team_input.lower() == 'quit':
                break
                
            if not team_input:
                print("Please enter at least one Loomian\n")
                continue
                
            processed_team = []
            invalid_loomians = []
            
            for mon in [mon.strip() for mon in team_input.split(',')]:
                lower_mon = mon.lower()
                if lower_mon in loomian_lower_map:
                    processed_team.append(loomian_lower_map[lower_mon])
                else:
                    invalid_loomians.append(mon)
            
            if invalid_loomians:
                print(f"These Loomians are not recognized: {', '.join(invalid_loomians)}")
                print("Valid Loomians are:", ', '.join(sorted(loomian_lower_map.values())))
                raise ValueError("Invalid Loomian(s)")
            
            rank_input = input("Enter competitive rank (NOVICE, ADVANCED, HYPER, EXPERT, ACE): ").strip()
            if rank_input.lower() == 'quit':
                break
                
            lower_rank = rank_input.lower()
            if lower_rank not in rank_lower_map:
                print(f"Invalid rank. Must be one of: {', '.join(rank_encoder.keys())}")
                continue
                
            corrected_rank = rank_lower_map[lower_rank]
            
            completion = predict_team(processed_team, corrected_rank, model, tokenizer, rank_encoder)
            
            print("\n" + "=" * 50)
            print(f"Starting Team: {', '.join(processed_team)}")
            print(f"Competitive Rank: {corrected_rank}")
            print("-" * 50)
            print("Suggested Completion:")
            for i, mon in enumerate(completion[len(processed_team):], start=len(processed_team)+1):
                print(f"Slot {i}: {mon}")
            print("=" * 50 + "\n")
            
        except ValueError as ve:
            print(f"Input Error: {ve}\n")
        except Exception as e:
            print(f"An error occurred: {e}\nPlease try again.\n")

interactive_predictor_case_insensitive(model, tokenizer, rank_encoder)