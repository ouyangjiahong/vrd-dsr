#from __future__ import absolute_import

import os
import os.path as osp
import sys
import time
import pdb
import cPickle
import argparse
from glob import glob

import torch
import torch.nn as nn
import torch.nn.init
import cv2
import numpy as np

import sys
import os.path as osp

from lib.nets.Vrd_Model import Vrd_Model
import lib.network as network
from lib.data_layers.vrd_data_layer import VrdDataLayer
from lib.model import test_pre_net, test_rel_net
from lib.blob import prep_im_for_blob
from easydict import EasyDict

class vrd_module():

    def __init__(self):
        self.args = EasyDict()
        self.args.dataset = 'vrd'
        self.args.use_so = True
        self.args.use_obj = True
        self.args.no_obj_prior = True
        self.args.loc_type = 0
        self.args.num_relations = 70
        self.args.num_classes = 100 # add background
        with open('data/vrd/so_prior.pkl', 'rb') as fid:
            self.so_prior = cPickle.load(fid)
        with open('data/vrd/obj.txt') as f:
            self.vrd_classes = [ x.strip() for x in f.readlines() ]
        with open('data/vrd/rel.txt') as f:
            self.vrd_rels = [ x.strip() for x in f.readlines() ]
        # Model
        self.net = Vrd_Model(self.args)
        self.net.cuda()
        self.net.eval()
        model_path = 'models/epoch_4_checkpoint.pth.tar'
        if osp.isfile(model_path):
            print("=> loading model '{}'".format(model_path))
            checkpoint = torch.load(model_path)
            self.net.load_state_dict(checkpoint['state_dict'])
        else:
            print "=> no model found at '{}'".format(args.resume)

    def relation_im(self, im_path, res):
        # pdb.set_trace()
        # draw detection image
        self.plot_detection_res(im_path, res)
        im_name = im_path.split('/')[-1]
        im_name = im_name[:-4]

        boxes_img = res['box']
        pred_cls_img = np.array(res['cls'])
        pred_confs = np.array(res['confs'])
        time1 = time.time()
        im = cv2.imread(im_path)
        ih = im.shape[0]
        iw = im.shape[1]
        PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
        image_blob, im_scale = prep_im_for_blob(im, PIXEL_MEANS)
        blob = np.zeros((1,)+image_blob.shape, dtype=np.float32)
        blob[0] = image_blob
        # Reshape net's input blobs
        boxes = np.zeros((boxes_img.shape[0], 5))
        boxes[:, 1:5] = boxes_img * im_scale
        classes = pred_cls_img
        ix1 = []
        ix2 = []
        n_rel_inst = len(pred_cls_img)*(len(pred_cls_img)-1)
        rel_boxes = np.zeros((n_rel_inst, 5))
        SpatialFea = np.zeros((n_rel_inst, 8))
        rel_so_prior = np.zeros((n_rel_inst, 70))
        i_rel_inst = 0
        for s_idx in range(len(pred_cls_img)):
            for o_idx in range(len(pred_cls_img)):
                if(s_idx == o_idx):
                    continue
                ix1.append(s_idx)
                ix2.append(o_idx)
                sBBox = boxes_img[s_idx]
                oBBox = boxes_img[o_idx]
                rBBox = self.getUnionBBox(sBBox, oBBox, ih, iw)
                rel_boxes[i_rel_inst, 1:5] = np.array(rBBox) * im_scale
                SpatialFea[i_rel_inst] = self.getRelativeLoc(sBBox, oBBox)
                rel_so_prior[i_rel_inst] = self.so_prior[classes[s_idx], classes[o_idx]]
                i_rel_inst += 1

        if i_rel_inst == 0:
            # no relationship, return, or cause error
            f = open('test/' + im_name + '.txt', 'w')
            f.close()
            return

        boxes = boxes.astype(np.float32, copy=False)
        classes = classes.astype(np.float32, copy=False)
        ix1 = np.array(ix1)
        ix2 = np.array(ix2)
        # pdb.set_trace()
        obj_score, rel_score = self.net(blob, boxes, rel_boxes, SpatialFea, classes, ix1, ix2, self.args)
        rel_prob = rel_score.data.cpu().numpy()
        rel_prob += np.log(0.5*(rel_so_prior+1.0/self.args.num_relations))
        rlp_labels_im  = np.zeros((rel_prob.shape[0]*rel_prob.shape[1], 5), dtype = np.int)
        tuple_confs_im = []
        n_idx = 0
        for tuple_idx in range(rel_prob.shape[0]):
            sub = ix1[tuple_idx]
            obj = ix2[tuple_idx]
            for rel in range(rel_prob.shape[1]):
                conf = rel_prob[tuple_idx, rel]
                rlp_labels_im[n_idx] = [classes[sub], sub, rel, classes[obj], obj]
                tuple_confs_im.append(conf)
                n_idx += 1
        tuple_confs_im = np.array(tuple_confs_im)
        idx_order = tuple_confs_im.argsort()[::-1][:20]
        rlp_labels_im = rlp_labels_im[idx_order,:]
        tuple_confs_im = tuple_confs_im[idx_order]
        vrd_res = []

        # save test image + vrd res
        f = open('test/' + im_name + '.txt', 'w')

        for tuple_idx in range(rlp_labels_im.shape[0]):
            label_tuple = rlp_labels_im[tuple_idx]
            sub_cls = self.vrd_classes[label_tuple[0]]
            obj_cls = self.vrd_classes[label_tuple[3]]
            rel_cls = self.vrd_rels[label_tuple[2]]
            vrd_res.append(('%s%d-%s-%s%d'%(sub_cls, label_tuple[1], rel_cls, obj_cls, label_tuple[4]), tuple_confs_im[tuple_idx]))
            f.write(sub_cls + ',' + rel_cls + ',' + obj_cls + '\n')
        f.close()

        print vrd_res
        time2 = time.time()
        print "TEST Time:%s" % (time.strftime('%H:%M:%S', time.gmtime(int(time2 - time1))))
        return vrd_res

    def getUnionBBox(self, aBB, bBB, ih, iw, margin = 10):
            return [max(0, min(aBB[0], bBB[0]) - margin), \
                max(0, min(aBB[1], bBB[1]) - margin), \
                min(iw, max(aBB[2], bBB[2]) + margin), \
                min(ih, max(aBB[3], bBB[3]) + margin)]

    def getRelativeLoc(self, aBB, bBB):
        sx1, sy1, sx2, sy2 = aBB.astype(np.float32)
        ox1, oy1, ox2, oy2 = bBB.astype(np.float32)
        sw, sh, ow, oh = sx2-sx1, sy2-sy1, ox2-ox1, oy2-oy1
        xy = np.array([(sx1-ox1)/ow, (sy1-oy1)/oh, (ox1-sx1)/sw, (oy1-sy1)/sh])
        wh = np.log(np.array([sw/ow, sh/oh, ow/sw, oh/sh]))
        return np.hstack((xy, wh))

    def plot_detection_res(self, im_path, det_res):
        im_name = im_path.split('/')[-1]
        im_name = im_name[:-4]
        img = cv2.imread(im_path)
        for i in range(len(det_res['box'])):
            bbox = det_res['box'][i]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(img, self.vrd_classes[det_res['cls'][i]], (max(0,int(bbox[0])-5), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imwrite('test/'+im_name+'.jpg', img)

def pre_demo():
    vrdet = vrd_module()

    with open('data/vrd/test.pkl', 'rb') as fid:
        anno = cPickle.load(fid)

    with open('data/vrd/proposal.pkl', 'rb') as fid:
        proposals = cPickle.load(fid)

    anno_img = anno[0]
    im_path = 'img/3845770407_1a8cd41230_b.jpg'
    print im_path
    res = {}
    res['box'] = proposals['boxes'][0]
    res['cls'] = proposals['cls'][0]
    res['confs'] = proposals['confs'][0]
    print vrdet.relation_im(im_path, res)

def vrd_demo():
    from detector import detector
    im_path = 'img/3845770407_1a8cd41230_b.jpg'
    det = detector()
    vrdet = vrd_module()
    det_res = det.det_im(im_path)
    vrd_res = vrdet.relation_im(im_path, det_res)
    # print vrd_res

def vrd_demo_test_all():
    from detector import detector
    # build models
    det = detector()
    vrdet = vrd_module()

    # load all test image path
    img_paths = glob('data/sg_dataset/sg_test_images/*.jpg')
    img_paths.sort()
    # pdb.set_trace()
    # img_test_save_path = 'test/'

    # test each image
    for im_path in img_paths:
        print('start testing ' + im_path)
        det_res = det.det_im(im_path)
        vrd_res = vrdet.relation_im(im_path, det_res)
        print('finish testing ' + im_path)


if __name__ == '__main__':
    # vrd_demo()
    vrd_demo_test_all()
    #from IPython import embed; embed()
