#! /usr/bin/python env
#coding: utf-8

import os
import sys
import datetime
import time
import numpy as np
import logging
import json
import collections

import MySQLdb #this project develop with python2, so import MySQLdb, not pymysql as in python3
import oss2   #图片存储
import tensorflow as tf
from tensorflow.python.platform import gfile

WORKING_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"./")) #设置项目工作目录：当前文件所在目录

## 目录配置:四个重要目录:项目目录PROJECT_DIR，资源目录 RESOURCE_DIR，输入目录INPUT_DIR，输出目录OUTPUT_DIR，理论上可以互相独立
PROJECT_DIR = os.path.join(WORKING_DIR,"../")  #工程项目目录，主要是程序文件
RESOURCE_DIR= PROJECT_DIR + './resources/'  #资源目录，模型数据，初始化配置文件之类
INPUT_DIR = PROJECT_DIR     #输入根目录，输入数据的存储目录，因数据量比较大，所以输入目录不一定在工程目录之下
OUTPUT_DIR = PROJECT_DIR + './output'   #输出根目录，输出数据的存储目录，因数据量比较大，所以输出目录不一定在工程目录之下

sys.path.insert(0,PROJECT_DIR) #插入搜索路径最前面
##----------------------------------------------------------------------------------

## 日志系统配置
logger = logging.getLogger('imagevec_logger')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()    # 写入到控制台
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s-%(filename)s-%(lineno)d:    %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
##-----------------------------------------------------------------------------

## mysql 属性数据库配置
#mysql_host = 'rm-bp1iw6jmy10og8w5r.mysql.rds.aliyuncs.com'   # 内网地址，远程主机的ip地址
mysql_host = 'rm-bp1iw6jmy10og8w5ruo.mysql.rds.aliyuncs.com' # 外网地址，远程主机的ip地址
mysql_user = 'tujing_read'    # MySQL用户名
mysql_passwd = 'sl677JsYs'    # 用户密码
mysql_db = 'tujing'           # database名
mysql_port = int(3306)          #数据库监听端口，默认3306，整型数据
mysql_charset = 'utf8'           #指定utf8编码的连接
##---------------------------------------------------------------------------------------

## oss 图片数据库配置
oss_access_key_id = os.getenv('OSS_TEST_ACCESS_KEY_ID', 'LTAIOhsKagLIaskD')
oss_access_key_secret = os.getenv('OSS_TEST_ACCESS_KEY_SECRET', 'b507nOLWAwwfwIB4l2cEAcZCyJ3Q5h')
oss_bucket_name = os.getenv('OSS_TEST_BUCKET', 'tujing-clothimage') #  Bucket = tujing-clothimage
#oss_bucket_name = os.getenv('OSS_TEST_BUCKET', 'tujing-clothimage-search') # 搜索 Bucket = tujing-clothimage-search

#oss_endpoint = os.getenv('OSS_TEST_ENDPOINT', 'oss-cn-hangzhou-internal.aliyuncs.com') #内网 ECS 的经典网络访问域名  http://tujing-clothimage.oss-cn-hangzhou-internal.aliyuncs.com
oss_endpoint = os.getenv('OSS_TEST_ENDPOINT', 'oss-cn-hangzhou.aliyuncs.com') #外网访问域名 http://tujing-clothimage.oss-cn-hangzhou.aliyuncs.com
oss_bucket = oss2.Bucket(oss2.Auth(oss_access_key_id, oss_access_key_secret), oss_endpoint, oss_bucket_name)# 创建Bucket对象，所有Object相关的接口都可以通过Bucket对象来进行
##---------------------------------------------------------------------------------------------------------------

## tf生成imagevector配置
INCEPTIONV3_MODEL_DIR = RESOURCE_DIR + './inception_v3/' # 下载的谷歌训练好的inception-v3模型文件目录
INCEPTIONV3_MODEL_FILE = 'classify_image_graph_def.pb' # 下载的谷歌训练好的inception-v3模型文件名
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'    # inception-v3 模型中代表瓶颈层结果的张量名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0' # 图像输入张量所对应的名称

OFFLINE_UPDATE_PATH = os.path.join(OUTPUT_DIR,'offline_update.json')    #离线更新日志 绝对路径
OFFLINE_CONFIG_PATH = os.path.join(OUTPUT_DIR,'offline_config.json')    #离线向量配置 绝对路径
# INPUT_DATA = PROJECT_DIR + './data/test_data/' # 图片数据的文件夹


##-----------------------------------------------------------------------------------------------

# # 获取图片list
# def read_image_list():
#     # 获取当前目录下所有的有效图片文件
#     extensions = ['jpg', 'jepg', 'JPG', 'JPEG']
#     file_list = []
#     dir_name = os.path.basename(INPUT_DATA)
#     for extension in extensions:
#         file_glob = os.path.join(INPUT_DATA, '*.' + extension)  # 将多个路径组合后返回
#         file_list.extend(glob.glob(file_glob))  # glob.glob返回所有匹配的文件路径列表，extend往列表中追加另一个列表
#
#     # image_list 仅保存文件的名字, 如 20180524_IMG_5248.JPG
#     image_list = []
#     for file_name in file_list:  # 遍历此类图片的每张图片的路径
#         base_name = os.path.basename(file_name)  # 路径的基本名称也就是图片的名称，如：102841525_bd6628ae3c.jpg
#         image_list.append(base_name)
#
#     return image_list

def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

def get_bottlenecks(sess, img_key,img_save_path, jpeg_data_tensor, bottleneck_tensor,BOTTLENECK_DIR):

    bottleneck_path = os.path.join(BOTTLENECK_DIR,'{img_key}.txt'.format(img_key=img_key)) #注意
    logger.info("To Write img " + img_key + " vector to " + bottleneck_path)
    sys.stdout.flush()
    if os.path.exists(bottleneck_path):
        return

    image_data = gfile.FastGFile(img_save_path, 'rb').read()
    bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)

    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_string = ",".join(str(x) for x in bottleneck_values)
        bottleneck_file.write(bottleneck_string);

def RankerBaseline(img_dis, img_name, req):
    ## mysql 属性数据库配置
    # mysql_host = 'rm-bp1iw6jmy10og8w5r.mysql.rds.aliyuncs.com'   # 内网地址，远程主机的ip地址
    mysql_host = 'rm-bp1iw6jmy10og8w5ruo.mysql.rds.aliyuncs.com'  # 外网地址，远程主机的ip地址
    mysql_user = 'tujing_read'  # MySQL用户名
    mysql_passwd = 'sl677JsYs'  # 用户密码
    mysql_db = 'tujing'  # database名
    mysql_port = int(3306)  # 数据库监听端口，默认3306，整型数据
    mysql_charset = 'utf8'  # 指定utf8编码的连接
    mysql_conn = MySQLdb.connect(mysql_host, mysql_user, mysql_passwd, mysql_db, mysql_port, mysql_charset)
    mysql_cursor = mysql_conn.cursor()  # 创建一个光标，然后通过光标执行sql语句

    prod_name=[]
    prod_dis=[]
    for row in range(0,len(img_name)): #每一row 代表 一个图片的搜索结果
        sample_dises_dict={}  #采用字典{ sample_id : [dis1,dis2,...] }，相同sample_id聚合
        for col in range(0,len(img_name[row])):
            id = img_name[row][col]
            mysql_cursor.execute( ' select sample_id from tujing.sample_imgs  where tujing.sample_imgs.img_id= {0} '.format(id))
            sample_id = mysql_cursor.fetchall()  # 取出cursor得到的数据
            try:
                key = sample_id[0][0]           # 取出 img_id 对应的 sample_id ，此处 sample_id 结果是个二维数组
                # key不存在，则新建，若已存在，则追加
                if key not in sample_dises_dict:
                    value = []
                    value.append(img_dis[row][col])
                    sample_dises_dict[key] = value
                else:
                    sample_dises_dict[key].append(img_dis[row][col])
            except: Exception,"sample_id error"

        #本行 kv 聚合结束 sample_dises_dict = {sample_id: [dis1, dis2, ...]}
        dis_name=[]
        for id in sample_dises_dict:
            try: #计算 sample_id对应的平均距离
                mean_dis = sum(sample_dises_dict[id])/len(sample_dises_dict[id])
                dis_name.append( (mean_dis,id) )
            except: Exception,"divided by zero"

        ## 单行 均值距离计算完成
        disStyle_name = []  #考虑 风格之后的 搜索距离-样品 列表
        ALL_SELECTED_CODE = 0 # 风格 全选时的 style_id 取值
        STYLE_DISTANCE = 10000 #无任何风格被选中 时的 距离加成
        for x in range(0,len(dis_name)):
            #mysql_cursor.execute( ' select * from tujing.sample_styles join ( select sample_id from tujing.sample_imgs  where tujing.sample_imgs.img_id= {0} ) as A on A.sample_id= tujing.sample_styles.sample_id '.format(id))
            id=dis_name[x][1]
            mysql_cursor.execute( ' select style_id from tujing.sample_styles  where tujing.sample_styles.sample_id= {0} '.format(id))
            style_id = mysql_cursor.fetchall()  # 取出cursor得到的数据
            for i in range(0,len(style_id)):  # style_id[][] 与 req 的 档位关系, sytle_id 来自 搜索库，req来自请求，应该采用 搜索结果 匹配 req
                style = style_id[i][0] #取出 sample_style 表中 的风格代码
                styleList_req = req[row]  #取出查询数据中 选中的 风格列表
                if (style in styleList_req) or (ALL_SELECTED_CODE in styleList_req):
                    disStyle_name.append(dis_name[x]) #若 搜索结果风格 包含在 请求风格列表中 或者 请求列表风格全选，则保持 距离不变
                else:
                    disStyle_name.append( (dis_name[x][0] + STYLE_DISTANCE,dis_name[x][1]) ) #风格不符合，则 搜素距离 加长

        # 考虑 风格之后的 搜索距离-样品 列表 disStyle_name 按照 搜索距离排序
        disStyle_name_sorted = sorted(disStyle_name)

        #将 排序后的 dis-name 分别存到 本行的 列表中
        row_prod_name = [] #存放单张搜索结果 的sample_id
        row_prod_dis = []  #存放单张搜索结果 sample_id 对应 的搜索距离
        for x in disStyle_name_sorted:
            row_prod_dis.append(x[0])
            row_prod_name.append(x[1])

        prod_name.append(row_prod_name) #将每一行即 单张 的搜索结果 加入 总体搜索结果
        prod_dis.append(row_prod_dis)

    #得到 prod_name[][] prod_dis[][]
    return  prod_name, prod_dis


def main(_):
    #image_list = read_image_list();

    #======================================================================================================
    img_name = [[1179, 1180,   1182, 1183, 1184],  # img_id 1179,1180 = sample_id 271 = style_id 4, img_id 1182,1183,1184 = sample_id 272 = style_id 4
                [1179, 1180,   1182, 1183, 1184],
                [1179, 1180,   1182, 1183, 1184],
                [1179, 1180,   1182, 1183, 1184],
                [1179, 1180,   1182, 1183, 1184],
                [1179, 1180,   1182, 1183, 1184]]  #<type 'tuple'>: (508L, 271L, 4, datetime.datetime(2018, 8, 23, 7, 5, 40), datetime.datetime(2018, 8, 23, 7, 5, 40))
                                             #<type 'tuple'>: ((509L, 272L, 4, datetime.datetime(2018, 8, 23, 7, 9, 4), datetime.datetime(2018, 8, 23, 7, 9, 4)),)
    img_dis = [[0.1, 0.3, 0.1, 0.2, 0.6],   # 按照原始搜索结果 271, 272
               [0.1, 0.3, 0.1, 0.2, 0.1],   # 按照原始搜索结果 272, 271
               [0.1, 0.3, 0.1, 0.2, 0.6],   # 按照原始搜索结果 271, 272
               [0.1, 0.3, 0.1, 0.2, 0.1],   # 按照原始搜索结果 272, 271
               [0.1, 0.3, 0.1, 0.2, 0.6],   # 按照原始搜索结果 271, 272
               [0.1, 0.3, 0.1, 0.2, 0.1],]  # 按照原始搜索结果 272, 271
    req = [[2, 3], #无风格被选中，距离全部加长，但与原始搜索结果相比，排序不变
           [2, 3],
           [0],   #默认分割全选，搜索距离不变，排序不变
           [0],
           [1, 4], #风格4被选中，搜索距离会受风格档位的影响
           [1, 4]]
    # TODO: 排序
    prod_name, prod_dis = RankerBaseline(img_dis, img_name, req)
    #======================================================================================================

   # 读取模型
    with gfile.FastGFile(os.path.join(INCEPTIONV3_MODEL_DIR, INCEPTIONV3_MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 加载模型，返回对应名称的张量
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECK_TENSOR_NAME,
                                                                                          JPEG_DATA_TENSOR_NAME])

    #-------------------------------------------------------------------------------------------------------------
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        mysql_conn = MySQLdb.connect(mysql_host, mysql_user, mysql_passwd, mysql_db, mysql_port, mysql_charset)
        mysql_cursor = mysql_conn.cursor()  # 创建一个光标，然后通过光标执行sql语句
        mysql_cursor.execute('select distinct comp_id from tujing.sample_imgs ')
        comp_id = mysql_cursor.fetchall()  # 取出cursor得到的数据
        mysql_cursor.execute('select distinct craft_id from tujing.sample_imgs ')
        craft_id = mysql_cursor.fetchall()  # 取出cursor得到的数据
        OfflineConfig_dict = collections.OrderedDict()
        OfflineConfig_dict["index_info"] = {"comp_cnt": 0, "index_cnt": 0}
        index_cnt = 0

        # 判断是否需要 更新
        UPDATE_FLAG = False

        if not os.path.exists(OFFLINE_UPDATE_PATH):  #首先判断是否存在文件，如果不存在，则直接赋值0000-00-00 00:00:00
            offline_update = {'offline_update_time':'1949-10-01 00:00:00'}
        else: #文件存在，则在文件中读取 更新配置
            with open(OFFLINE_UPDATE_PATH, 'r') as readUpdate:
                offline_update = json.load(readUpdate)
        assert offline_update is not None
        LAST_OFFLINE_UPDATE = offline_update['offline_update_time'].encode("utf-8")

        for comp in comp_id:
            mysql_cursor.execute( #筛选 comp_id == comp 中 updateAt最大的 数据
                'SELECT * FROM tujing.sample_imgs '
                'WHERE comp_id={0} AND date_format(updatedAt,"%Y-%m-%d %H:%i:%s") >"{1}"'.format(int(comp[0]),LAST_OFFLINE_UPDATE)
            )
            updateAt = mysql_cursor.fetchall()  # 取出cursor得到的数据
            if len(updateAt) > 0: #自上次 更新索引以来，数据更新的数量
                UPDATE_FLAG = True

        if UPDATE_FLAG: #若有更新，则生成新的索引
            for comp in comp_id:
                for craft in craft_id:
                    #生成 离线配置文件
                    index_cnt +=1

                    # 临时图片保存目录按公司，公司之下再分工艺， 保存后用于提取特征向量
                    IMAGE_DIR = os.path.join(PROJECT_DIR,
                                             'tmp/{comp_id}/{craft_id}'.format(comp_id=comp[0], craft_id=craft[0]))
                    if not os.path.isdir(IMAGE_DIR):
                        os.makedirs(IMAGE_DIR)
                    assert os.path.isdir(IMAGE_DIR), 'Offline IMAGE_DIR directory create failure!'   # 对图片保存路径断言

                    # 向量保存目录按公司，公司之下再分工艺， 保存训练数据通过瓶颈层后提取的特征向量
                    BOTTLENECK_DIR = os.path.join(OUTPUT_DIR,
                                              'v3_imagevector/{comp_id}/{craft_id}'.format(comp_id=comp[0], craft_id=craft[0]))
                    if not os.path.isdir(BOTTLENECK_DIR):
                        os.makedirs(BOTTLENECK_DIR)
                    assert os.path.isdir(BOTTLENECK_DIR),'Offline vector directory create failure!' #对向量保存路径断言

                    item_dict = collections.OrderedDict()
                    item_dict["comp_id"] = comp[0]
                    item_dict["craft_id"] = craft[0]
                    item_dict["dim"] = 2048
                    item_dict["vector_dir_home"] = BOTTLENECK_DIR
                    OfflineConfig_dict["{}".format(index_cnt)]=item_dict

                    mysql_cursor.execute('select * from tujing.sample_imgs where comp_id = {0} and craft_id ={1}'.format(int(comp[0]),int(craft[0])))
                    value = mysql_cursor.fetchall()  # 取出cursor得到的数据
                    saveURL = open(os.path.join(IMAGE_DIR, "mysql_url.dat"), 'w') #保存mysql查询结果
                    for item in value:
                        #if item[8] < NOW:  #增量更新相关设置
                            saveURL.write(item[3] + '\n')
                            oss_key = item[3]
                            oss_fileName = item[3] # oss图片下载到本地文件，名称保持不变
                            oss_imgSavePath = os.path.join(IMAGE_DIR, oss_fileName)
                            oss_result = oss_bucket.get_object_to_file(oss_key, oss_imgSavePath)  # oss图片下载到本地文件

                            get_bottlenecks(sess, oss_key,oss_imgSavePath, jpeg_data_tensor, bottleneck_tensor,BOTTLENECK_DIR)

                    saveURL.close()

            OfflineConfig_dict["index_info"] = {"comp_cnt": len(comp_id), "index_cnt": index_cnt}

            # 保存离线处理 配置输出，提供给搜索
            with open(OFFLINE_CONFIG_PATH, 'w')  as saveConfig:
                saveConfig.write(json.dumps(OfflineConfig_dict))
                saveConfig.close() #关闭离线配置文件

            # 保存离线更新时间，用作下次巡检时判断是否需要 重新 生成索引向量
            with open(OFFLINE_UPDATE_PATH, 'w')  as saveUpdate:
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") #此处与date_format表述不同但结果要相同
                update_dict={'offline_update_time':now}
                json.dump(update_dict,saveUpdate)

        else: #此次巡检发现无更新
            pass

        mysql_cursor.close()
        mysql_conn.close()  # 最后记得关闭光标和连接，防止数据泄露

    #====================== 生成周期性巡检完成标志文件 ======================================
    checked_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checked_flag_filename = checked_time + '_DONE'
    CHECKED_FLAG_PATH = os.path.join(OUTPUT_DIR,
                                  '{0}'.format(checked_flag_filename))
    os.system(r'touch %s' % CHECKED_FLAG_PATH)

if __name__ == '__main__':
    tf.app.run()

