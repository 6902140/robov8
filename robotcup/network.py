from enum import Enum

class DataType(Enum):
    TEAM_ID = 0
    RESULT = 1
    REQUEST_ROT = 3

def pack_data(type: DataType, text: str) -> bytearray:
    packed = bytearray()
    packed.extend(type.value.to_bytes(4, 'big'))
    data = text.encode("utf-8")
    length = len(data).to_bytes(4, 'big')
    packed.extend(length)
    packed.extend(data)
    return packed
