import argparse


import config


class PlayGrounds:

    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('-env', help='Environments to work with '
                                         'Options: keras, tensorflow, opencv', required=True)
        parser.add_argument('-feature', help="Select Features to execute", required=True)
        parser.add_argument('-worker', help="Select Algorithm to execute", required=True)
        parser.add_argument('-input', help="Select Input to feed to network")
        parser.add_argument('-type', help="Select Input to feed to network")
        parser.add_argument('-ftype', help="Select Input type(video, image) to feed to network")

        args = parser.parse_args()

        self.env = args.env
        self.feature = args.feature
        self.worker = args.worker
        self.input = args.input
        self.type = args.type
        self.inputType = args.ftype


    def runScript(self):
        assert self.env in config.env, "Unsupported env please select another one"
        env = config.env[self.env]
        assert self.feature not in env, "Unsupported feature please select another one"
        feature = env['features'][self.feature]


        if self.type == "train":
            feature.trainFeature(self.worker)
        if self.type == "predict":
            feature.runFeature(self.worker, self.input, self.inputType)



if __name__ == '__main__':
    run = PlayGrounds()
    run.runScript()
