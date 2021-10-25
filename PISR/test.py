from models.self_adder_fsrcnn import SelfAdderFSRCNNStudent
from models.conv_vdsr import VDSRTeacher, ParallelVDSRTeacher


if __name__ == '__main__':
    # test = SelfAdderFSRCNNStudent(2, 1)
    # for k, v in test.state_dict().items():
    #     print(k)

    test = VDSRTeacher(2, 1)
    for k, v in test.state_dict().items():
        print(k)
    print('='*50)
    test = ParallelVDSRTeacher(2, 1)
    for k, v in test.state_dict().items():
        print(k)