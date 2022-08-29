from data.dataLoader import setDataLoader
from operation.imagePreProcess import Preprocess
import torch
from model.encoder import ReferenceEncoder , ContentEncoder
from model.decoder import Decoder
from model.loss import reconLoss , harmonizedLoss , distillLoss
from operation.imagePreProcess import Tensor2PIL


class OPT():
    pass

opt = OPT()
opt.norm = "BN"
opt.preprocess = "none"
opt.no_flip = True

def Train():

    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    refEncoderA = ReferenceEncoder(4, 3, 32, 512, norm="BN", activ="relu", pad_type='reflect')
    refEncoderB = ReferenceEncoder(4, 3, 32, 512, norm="BN", activ="relu", pad_type='reflect')
    contEncoder = ContentEncoder(opt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    refEncoderA = refEncoderA.to(device)
    refEncoderB = refEncoderB.to(device)
    contEncoder = contEncoder.to(device)
    fusionDecoderA = Decoder(opt)
    fusionDecoderB = Decoder(opt)
    fusionDecoderA = fusionDecoderA.to(device)
    fusionDecoderB = fusionDecoderB.to(device)
    dataLoader = setDataLoader("C:\workspace\SSIH\dataset\lite")


    contOptim = torch.optim.Adam(contEncoder.parameters() , lr=2e-4)
    refAOptim = torch.optim.Adam(refEncoderA.parameters(), lr=2e-4)
    refBOptim = torch.optim.Adam(refEncoderB.parameters(), lr=2e-4)
    decAoptim = torch.optim.Adam(fusionDecoderA.parameters(), lr=2e-4)
    decBoptim = torch.optim.Adam(fusionDecoderB.parameters(), lr=2e-4)

    for epoch in range(100):
        print('SSIH Training Epochs : ', epoch)

        for i, x in enumerate(dataLoader):
            contentA, referenceA , contentB , referenceB = Preprocess(x)

            contOptim.zero_grad()
            refAOptim.zero_grad()
            refBOptim.zero_grad()
            decAoptim.zero_grad()
            decBoptim.zero_grad()

            contentA = contentA.to(device)
            referenceA = referenceA.to(device)
            contentB = contentB.to(device)
            referenceB = referenceB.to(device)

            contEncoder.train()
            refEncoderA.train()
            refEncoderB.train()
            fusionDecoderA.train()
            fusionDecoderB.train()

            contentBFV, _, _, _, _, _ = contEncoder(contentB)
            contentAFV , conv , pad_left , pad_right , pad_top , pad_bottom = contEncoder(contentA)

            referenceAFV = refEncoderA(referenceA)
            contentReferenceAFV = refEncoderA(referenceA)
            referenceBFV = refEncoderB(referenceB)
            reContentA = fusionDecoderA( contentAFV , referenceAFV , conv , pad_left, pad_right, pad_top, pad_bottom )
            reContentB = fusionDecoderB( contentAFV , referenceBFV , conv , pad_left, pad_right, pad_top, pad_bottom )

            totalLoss = reconLoss(contentA , reContentA) + harmonizedLoss(contentB , reContentB) + distillLoss( contentAFV , contentBFV , contentReferenceAFV , referenceAFV )
            print('Total Loss : ' , totalLoss.item() )
            totalLoss.backward()
            contOptim.step()
            refAOptim.step()
            refBOptim.step()
            decAoptim.step()
            decBoptim.step()

            if epoch%10 == 0 and i==0:
                contentA = Tensor2PIL(contentA[0])
                contentB = Tensor2PIL(contentB[0])
                recontentA = Tensor2PIL(reContentA[0])
                recontentB = Tensor2PIL(reContentB[0])
                contentA.show()
                contentB.show()
                recontentA.show()
                recontentB.show()







if __name__ == '__main__':
    Train()