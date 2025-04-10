import os
import tokenize

def remove_comments_from_file(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        tokens = tokenize.generate_tokens(infile.readline)
        prev_end = (1, 0)
        for toknum, tokval, start, end, line in tokens:
            if toknum == tokenize.COMMENT:
                continue
            if start > prev_end:
                outfile.write(' ' * (start[1] - prev_end[1]))
            outfile.write(tokval)
            prev_end = end

def process_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.py') and not filename.endswith('_nocomments.py'):
            input_path = os.path.join(directory, filename)
            output_path = os.path.join(directory, filename[:-3] + '_nocomments.py')
            remove_comments_from_file(input_path, output_path)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python remove_comments.py <directory>")
        sys.exit(1)
    process_directory(sys.argv[1])