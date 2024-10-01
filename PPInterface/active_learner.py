import torch
import torch.nn as nn
import torch.optim as optim
from esm import pretrained


class ESMAdaptor(nn.Module):
    def __init__(self, esm_model_name="esm2_t6_8M_UR50D", output_dim=1):
        super(ESMAdaptor, self).__init__()
        self.esm_model, _ = pretrained.load_model_and_alphabet_together(esm_model_name)
        for param in self.esm_model.parameters():
            param.requires_grad = False
        self.cls_head = nn.Linear(self.esm_model.embed_dim, output_dim)

    def forward(self, tokens):
        results = self.esm_model(tokens, repr_layers=[self.esm_model.num_layers])
        cls_rep = results["representations"][self.esm_model.num_layers][:, 0, :]
        output = self.cls_head(cls_rep)
        return output



def train_adaptor(sequences, labels, num_epochs=4, batch_size=1, learning_rate=1e-4):
    # Load small ESM model and alphabet
    esm_model_name = "esm2_t6_8M_UR50D"  # Small ESM model
    model = ESMAdaptor(esm_model_name, output_dim=1)
    model.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.cls_head.parameters(), lr=learning_rate)

    _, alphabet = pretrained.load_model_and_alphabet(esm_model_name)
    batch_converter = alphabet.get_batch_converter()
    dataset = list(zip(sequences, labels))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            batch_sequences, batch_targets = batch
            batch_labels = torch.tensor(batch_targets).float().unsqueeze(1)
            batch_tokens = batch_converter(batch_sequences)[2]
            predictions = model(batch_tokens)
            loss = criterion(predictions, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}')
    print("Training complete.")
    return model