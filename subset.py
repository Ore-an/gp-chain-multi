import argparse
from collections import defaultdict
from numpy import random

parser = argparse.ArgumentParser(description='Helper script to subset data')
subparsers = parser.add_subparsers(dest='subparser_name')
parser_c = subparsers.add_parser('check', help='Check the total duration of the utterances')
parser_s = subparsers.add_parser('subset', help='Subset the data')
parser_c.add_argument('-c', '-check', dest='check', action='store_true')
parser_s.add_argument('-hr', '--hours', dest='hours', nargs='?', default=5, type=int, help='The lenght in hours of the desired subset (default=5)')
parser_c.add_argument('input', type=str, help='Input file (wav-to-duration output)')
parser_s.add_argument('input', type=str, help='Input file (wav-to-duration output)')
parser_s.add_argument('output', type=str, help='Output file, a list of selected utterances')
args = parser.parse_args()

def check():
    with open(args.input, 'r') as f:
        uttlen = {}
        for line in f:
            uttlen[line.split()[0]] = float(line.split()[1])
        tot = sum(uttlen.values())
        print "%s length: %d h, %d min, %d s" % (args.input, (tot//3600), ((tot % 3600)//60), ((tot % 3600) % 60))
        return tot, uttlen

def subset():
    tot, uttlen = check()
    if args.hours > tot//3600:
        print('The dataset is smaller than the desired subset')
        raise SystemExit

    h = 3600 # one hour in seconds
    remlen = h*args.hours

    uttlist = []
    
    while True:
        utt = random.choice(uttlen.keys())
        uttlist.append(utt)
        remlen -= uttlen[utt]
        del uttlen[utt]
        if remlen <= 0:
            actlen = h*args.hours + (-remlen)
            print "Subset actual length: %d h, %d min, %d s" % ((actlen//3600), ((actlen % 3600)//60), ((actlen % 3600) % 60))
            break
        uttlist = sorted(uttlist)
        with open(args.output, 'w') as f:
            for utt in uttlist:
                f.write("{}\n".format(utt))


if args.subparser_name == 'check':
    check()
elif args.subparser_name == 'subset':
    subset()
else:
    raise ValueError('First argument has to be "check" or "subset"')
