from models.self_adder_fsrcnn import SelfAdderFSRCNNStudent


if __name__ == '__main__':
    test = SelfAdderFSRCNNStudent(2, 1)
    for k, v in test.state_dict().items():
        print(k)