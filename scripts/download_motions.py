import urllib2
import zipfile
import os

data_path = '../data/'
motions = []
for file_name in os.listdir(data_path):
    if not os.path.isdir(os.path.join(data_path, file_name)):
        continue
    split = file_name.split('files_motions_')
    if len(split) != 2:
        continue
    motion = split[1]
    motions.append(motion)

motions = ['515', '516', '517', '520', '521', '525', '526']
n_motions = len(motions)
print('Downloading %d motions ...' % n_motions)

output_dir = '../download/'
for idx, motion in enumerate(motions):
    url = 'https://motion-database.humanoids.kit.edu/file_download_archive/motions/%s/' % motion
    file_name = 'files_motions_%s' % motion
    zip_path = os.path.join(output_dir, file_name + '.zip')
    extract_path = os.path.join(output_dir, file_name)

    try:
        u = urllib2.urlopen(url)
    except:
        continue
    f = open(zip_path, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print('(%.2d/%d) downloading: %s (%s bytes)' % (idx+1, n_motions, file_name, file_size))

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print status,
    f.close()

    zf = zipfile.ZipFile(zip_path)
    zf.extractall(extract_path)

for motion in motions:
    file_name = 'files_motions_%s' % motion
    extract_path = os.path.join(output_dir, file_name)
    if os.path.isdir(extract_path):
        print('%s: okay' % file_name)
    else:
        print('%s: fail' % file_name)
