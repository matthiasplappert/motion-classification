import os

path = './'
for curr_dir in os.listdir(path):
	curr_dir_path = os.path.join(path, curr_dir)
	if not os.path.isdir(curr_dir):
		continue
	for curr_file in os.listdir(curr_dir_path):
		curr_file_path = os.path.join(curr_dir_path, curr_file)
		split = os.path.splitext(curr_file_path)
		if len(split) != 2 or split[1].lower() != '.xml':
			continue
		if curr_file.endswith('_nlopt.xml'):
			continue
		new_file_path = os.path.join(curr_dir_path, curr_file[:-4]) + '_nlopt.xml'
		print('renaming %s to %s' % (curr_file_path, new_file_path))
		os.rename(curr_file_path, new_file_path)
