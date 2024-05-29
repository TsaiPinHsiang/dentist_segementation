import os

if __name__ == '__main__':

    root = 'dataset'
    samdir = 'Sample'
    redir = 'result'

    imgdir = imgdir = list(sorted(os.listdir(os.path.join(root, samdir))))
    print(imgdir)

    try:
        os.mkdir(redir)
    except:
        pass

    for i in imgdir:
        try:
            det_tem = os.path.join(redir,  i)
            det_tem2 = os.path.join(redir, i, 'det')
            seg_tem2 = os.path.join(redir, i, 'seg')
            os.mkdir(det_tem)
            os.mkdir(det_tem2)
            os.mkdir(seg_tem2)
        except:
            pass
