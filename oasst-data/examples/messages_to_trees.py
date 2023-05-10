import argparse
from collections import OrderedDict

from oasst_data.reader import read_message_list, read_messages, read_oasst_jsonl
from oasst_data.writer import write_message_trees
from oasst_data.schemas import ExportMessageTree, ExportMessageNode


def parse_args():
    parser = argparse.ArgumentParser(description="messages_to_trees")
    parser.add_argument(
        "--input_file_name",
        type=str,
        help="path to input .jsonl or .jsonl.gz input file",
    )
    parser.add_argument(
        "--output_file_name",
        type=str,
        help="path to output .jsonl or .jsonl.gz file",
    )
    parser.add_argument("--exclude-nulls", action="store_true", default=False)
    args = parser.parse_args()
    return args


def main():
    """Read oasst flat messages from input file and generate a flat trees from them."""
    args = parse_args()


    print(f"reading: {args.input_file_name}")
    

    #List of trees containing the messages
    trees: list[ExportMessageTree] = [] 
    all_messages = read_message_list(args.input_file_name)
    
    assert len(all_messages) > 0,"Messages from read_message_list at line 34 contains nothing"
    tree_ids = list(set(m.message_tree_id for m in all_messages))
    assert len(tree_ids) > 1

    for id in tree_ids:
        #Turn into dicts for easy lookup
        #assert len(message_dicts) > 1,"Message list on line 45 is empty"
        for msg in all_messages:
            msg.replies = []
        messages = [msg.dict() for msg in all_messages if msg.message_tree_id == id]
        #message_dicts = [msg.dict() for msg in all_messages if msg.message_tree_id == id]

        message_dict = {}
        for message in messages:
            message_dict[message['message_id']] = message
        

    
        
        print("Amount of messages in tree :",len(messages))
        for message in messages:
            parent_id = message.get('parent_id')
            
            if parent_id:
                parent = message_dict.get(parent_id)
                #print("Parent",parent)
                if parent:
                    parent.setdefault('replies', []).append(message)
            
            
            
        tree = ExportMessageTree(
                message_tree_id=id,
                tree_state=messages[0].get('tree_state'),
                #Message id == message_tree_id
                prompt=message_dict.get(id) #Prompt starts with the root message node, which is the first one as it is pre sorted
            )
        
        messages.clear()
        if tree.prompt:
            trees.append(tree)
        print("This many trees as of right now", len(trees))
        
    assert len(trees) > 0,"No trees were formulated"
     
    print(f"writing: {args.output_file_name}")
    write_message_trees(args.output_file_name, trees,args.exclude_nulls)


if __name__ == "__main__":
    main()