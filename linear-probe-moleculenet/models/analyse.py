from load import load_model



# calculate the parameters of the model
def analyse_model(model):
    # calculate the number of parameters
    params = sum([p.numel() for p in model.parameters()])
    print(f"Number of parameters: {params}")
    # Million
    print(f"Number of parameters: {params/1000000}M")
    # calculate the number of trainable parameters
    params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Number of trainable parameters: {params}")
    # Million
    print(f"Number of trainable parameters: {params/1000000}M")


if __name__ == '__main__':
    # Grover
    # model_name = 'Grover'
    # model_name = 'MolCLR'
    # model_name = 'KCL'
    # model_name = 'Uni-Mol'
    # model_name = 'SciBERT'
    # model_name = 'ChemBERTa-10M-MTR'
    # model_name = 'CHEM-BERT'
    # model_name = 'KV-PLM'
    # model_name = 'MolT5'
    # model_name = 'DEN-ChemBERTa'
    # model_name = 'BERT'
    # model_name = 'PubMedBERT'
    # model_name = 'T5'
    # model_name = 'RoBERTa'
    # model_name = 'CLAMP'
    # model_name = 'Mole-BERT'
    # model_name = 'Grover-Base'
    # model_name = 'MoMu'
    # model_name = 'BioLinkBERT'
    model_name = 'MoMu-TE'

    model = load_model(model_name)
    print(f'{model_name}:')
    analyse_model(model)
