from models.self_adder_fsrcnn import SelfAdderFSRCNNStudent
from models.conv_vdsr import VDSRTeacher, VDSRStudent, DecreaseVDSRStudent
from models.fsrcnn import FSRCNNTeacher, FSRCNNStudent


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':


    test = VDSRStudent(2, 1, m=8)
    for k, v in test.state_dict().items():
        print(k)
    print(count_parameters(test))
    print('='*50)
    test = DecreaseVDSRStudent(2, 1, m=4)
    for k, v in test.state_dict().items():
        print(k)
    print(count_parameters(test))
