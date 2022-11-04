class FewGlueArgs:

    def __init__(self, args):
        self.task_name = args.task_name 
        self.data_dir = args.data_dir
        # TODO: add them to args
        self.train_examples = -1
        self.eval_examples = -1
        self.dev32_examples = -1
        self.eval_set = 'dev'

class DataWrapper:
    """ This class is used to load the fewglue dataset. 
    """

    def __init__(self, upper_args):
        self.args = FewGlueArgs(upper_args)
        
        from data_utils.task_processors import PROCESSORS, load_examples, DEV32_SET, TRAIN_SET, DEV_SET, TEST_SET, METRICS, DEFAULT_METRICS
        train_ex, eval_ex, dev32_ex = self.args.train_examples, self.args.eval_examples, self.args.dev32_examples
        train_ex_per_label, eval_ex_per_label, dev32_ex_per_label = None, None, None
        dataset = {}
        dataset["train"] = load_examples(
            self.args.task_name, 
            self.args.data_dir, 
            TRAIN_SET, 
            num_examples=train_ex, 
            num_examples_per_label=train_ex_per_label)

    
        dataset["test"] = load_examples(
            self.args.task_name, 
            self.args.data_dir, 
            DEV_SET, 
            num_examples=eval_ex, 
            num_examples_per_label=eval_ex_per_label)

        
        dataset["valid"] = load_examples(
            self.args.task_name, 
            self.args.data_dir, 
            DEV32_SET, 
            num_examples=dev32_ex, 
            num_examples_per_label=dev32_ex_per_label)


        self.dataset = dataset