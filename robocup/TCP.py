import os
import time
from socket import *

server_ip = '192.168.56.1'
server_port = 6666
# 请勿修改该类型
DataType_IDSEND = 0
DataType_3D = 1
BUFSIZE = 1024  # 设置缓存区的大小

class Protocol:
    """
    规定：
        数据包头部占4字节
        整型占4字节
        字符串长度位占2字节
        字符串不定长
    """

    def __init__(self, bs=None):

        if bs:
            self.bs = bytearray(bs)
        else:
            self.bs = bytearray(0)

    def get_int32(self):
        try:
            ret = self.bs[:4]
            self.bs = self.bs[4:]
            return int.from_bytes(ret, byteorder='little')
        except:
            raise Exception("数据异常！")

    def get_str(self):
        try:
            # 拿到字符串字节长度(字符串长度位2字节)
            # length = int.from_bytes(self.bs[:2], byteorder='little')
            # # 再拿字符串
            # ret = self.bs[2:length + 2]
            # # 删掉取出来的部分
            # self.bs = self.bs[2 + length:]
            ret = self.bs
            return ret.decode(encoding='utf8')
        except:
            raise Exception("数据异常！")

    def add_int32(self, val):
        bytes_val = bytearray(val.to_bytes(4, byteorder='big'))
        self.bs += bytes_val


    def add_str(self, val):
        bytes_val = bytearray(val.encode(encoding='utf8'))
        self.bs += bytes_val

    def get_pck_not_head(self):
        return self.bs

    def get_pck_has_head(self, type):
        bytes_pck_length = bytearray(len(self.bs).to_bytes(4, byteorder='big'))
        datatype = bytearray(type.to_bytes(4,byteorder='big')) #大端
        print(bytes(datatype+bytes_pck_length))
        return datatype+bytes_pck_length + self.bs


if __name__ == '__main__':
   # p = Protocol()
    # with open('result.txt', "r") as f:
     #   for line in f:
      #      p.add_str(line)
   # p =p.get_pck_has_head(DataType_3D)


    p1 = Protocol()
    p1.add_str("xjtu")
    p1= p1.get_pck_has_head(DataType_IDSEND)
    print(bytes(p1))


    tcp_client_socket = socket(AF_INET, SOCK_STREAM)
    # 目的信息

    # 链接服务器
    tcp_client_socket.connect((server_ip, server_port))

    tcp_client_socket.send(p1)

    time.sleep(2)
    print("lets go")

    # 关闭套接字
    tcp_client_socket.close()
    tcp_client_socket2 = socket(AF_INET, SOCK_STREAM)
    tcp_client_socket2.connect((server_ip, server_port))

    tcp_client_socket2.send(p)


    print("lets go")
    tcp_client_socket2.close()
