import torch
from models_mae_DEX import mae_DEX_vit_base


if __name__ == "__main__":
    input = 'mae_pretrain_DEX_base.pth'
    output = 'mae_DEX_base.pth'
    checkpoint = torch.load(input, map_location="cpu", weights_only=False)

    dias_arch = mae_DEX_vit_base()

    state_dict = checkpoint['model']

    a = list(state_dict.keys())
    b = list(dias_arch.state_dict().keys())
    
    for k in list(state_dict.keys()):
        if k.startswith('blocks.') and 'mlp' in k:
            ks = k.split('.')
            state_dict['blocks.' +
                       k[len(ks[0]+'.'):len(ks[0]+'.')+len(ks[1])] +
                       '.DIAS.director' + 
                       k[len(ks[0]+'.'+ks[1]+'.'+ks[2]):]] = state_dict[k]
            
            for i in range(dias_arch.num_actors_list[int(ks[1])]):
                state_dict['blocks.' +
                        k[len(ks[0]+'.'):len(ks[0]+'.')+len(ks[1])] +
                        '.DIAS.actors.' + str(i) +  
                        k[len(ks[0]+'.'+ks[1]+'.'+ks[2]):]] = state_dict[k]
            del state_dict[k]


    c = list(state_dict.keys())

    missing, unexpected = dias_arch.load_state_dict(state_dict, strict=False)
    print(missing, unexpected)
    torch.save({"state_dict": state_dict,}, output)
