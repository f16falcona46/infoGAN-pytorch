from model import *
from trainer import Trainer


def main():
    Discrete_Vars = 10
    Continuous_Vars = 2
    Noise_Vars = 62

    fe = FrontEnd()
    d = D()
    q = Q()
    g = G(Discrete_Vars + Continuous_Vars + Noise_Vars)

    for i in [fe, d, q, g]:
        i.cuda()
        i.apply(weights_init)

    trainer = Trainer(g, fe, d, q, Discrete_Vars, Continuous_Vars, Noise_Vars, 28, 28, 100)
    trainer.train()

if __name__ == "__main__":
    main()