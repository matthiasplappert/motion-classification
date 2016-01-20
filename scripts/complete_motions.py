from argparse import ArgumentParser
import os


def _color_str(text, color=None):
    # Source: https://www.siafoo.net/snippet/88
    if color == 'green':
        return '\033[0;32m%s\033[0;m' % text
    elif color == 'red':
        return '\033[0;31m%s\033[0;m' % text
    else:
        return text


parser = ArgumentParser()
parser.add_argument('mmmtools', help='path to the MMMMTools binaries')
parser.add_argument('--data-path', help='path to the data folder', default='../data/')
args = parser.parse_args()

dynamics_bin_path = os.path.join(args.mmmtools, 'MMMDynamicsCalculator')
if not os.path.isfile(dynamics_bin_path) or not os.access(dynamics_bin_path, os.X_OK):
    exit('"%s" does not exist or is not executable' % dynamics_bin_path)

completor_bin_path = os.path.join(args.mmmtools, 'XMLMotionCompleter')
if not os.path.isfile(completor_bin_path) or not os.access(completor_bin_path, os.X_OK):
    exit('"%s" does not exist or is not executable' % completor_bin_path)

# Find all motions
print('finding motion files ...')
data_path = args.data_path
motions = []
for file_name in os.listdir(data_path):
    curr_path = os.path.join(data_path, file_name)
    if not os.path.isdir(curr_path):
        continue
    split = file_name.split('files_')
    if len(split) != 2:
        continue
    for xml_candidate in os.listdir(curr_path):
        if os.path.splitext(xml_candidate)[1].lower() != '.xml':
            continue
        xml_path = os.path.join(curr_path, xml_candidate)
        motions.append(xml_path)
print('found %d motions\n' % len(motions))

# Complete motions
print('completing motions ...')
completor_errors = []
for idx, motion in enumerate(motions):
    print('(%.3d/%.3d) %s ...' % (idx+1, len(motions), motion))

    # Delete existing file (if necessary)
    split = os.path.splitext(motion)
    completed_path = ''.join([split[0], '_completed', split[1]])
    if os.path.exists(completed_path):
        os.remove(completed_path)

    cmd = ('%s --motion %s > /dev/null' % (completor_bin_path, motion))
    if os.system(cmd) == 0:
        if not os.path.exists(completed_path):
            # File should now exist, report error
            completor_errors.append(motion)
            print _color_str('fail\n', 'red')
        else:
            os.remove(motion)
            os.rename(completed_path, motion)
            print _color_str('okay\n', 'green')
    else:
        # Clean up (if necessary)
        if os.path.exists(completed_path):
            os.remove(completed_path)
        completor_errors.append(motion)
        print _color_str('fail\n', 'red')

# Calculate dynamics
print('calculating dynamics ...')
dynamics_errors = []
for idx, motion in enumerate(motions):
    print('(%.3d/%.3d) %s ...' % (idx+1, len(motions), motion))
    cmd = ('%s --inputMotion %s --outputMotion %s > /dev/null' % (dynamics_bin_path, motion, motion))
    if os.system(cmd) == 0:
        print _color_str('okay\n', 'green')
    else:
        dynamics_errors.append(motion)
        print _color_str('fail\n', 'red')

# Results
print('processed %d motions with %d completor errors and %d dynamics errors' % (len(motions), len(completor_errors), len(dynamics_errors)))
print('\ncompletor errors:')
if len(completor_errors) == 0:
    print('none')
else:
    for err in completor_errors:
        print(err)
print('\ndynamics errors:')
if len(dynamics_errors) == 0:
    print('none')
else:
    for err in dynamics_errors:
        print(err)
