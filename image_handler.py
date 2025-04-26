import os
import numpy as np
import face_recognition
import faiss
import io
import base64
from typing import List
from constant import EMBEDDING_FILE, FAISS_INDEX_FILE, FILENAME_FILE
from request_model.file_request import FileRequest
from model.stored_file import StoredFile
from db import insert_metadata, delete_metadata_batch

def insert_new_image(image_bytes, key: str, index, filenames, embeddings, metadata=None) -> dict:
    if not key.strip():
        return {"success": False, "message": "key is empty"}
    if _is_filename_exists(key):
        return {"success": False, "message": f"{key}key đã tồn tại"}
    # Đọc ảnh và trích xuất khuôn mặt
    try:
        image = face_recognition.load_image_file(io.BytesIO(image_bytes))
        face_encodings = face_recognition.face_encodings(image)
    except Exception as e:
        return {"success": False, "message": "Đã có lỗi xảy ra"}

    if not face_encodings:
        return {"success": False, "message": "⚠️ Không tìm thấy khuôn mặt trong ảnh."}

    new_embedding = face_encodings[0].astype("float32")

    # Cập nhật FAISS index
    index.add(np.array([new_embedding]))

    # Cập nhật embeddings & filenames
    embeddings = np.vstack([embeddings, new_embedding])
    filenames.append(key)

    # Ghi lại dữ liệu
    np.save(EMBEDDING_FILE, embeddings)
    with open(FILENAME_FILE, "w") as f:
        f.writelines([name + "\n" for name in filenames])
    faiss.write_index(index, FAISS_INDEX_FILE)

    # ✅ Lưu metadata vào SQLite
    if metadata:
        insert_metadata(key, metadata)
    return {"success": True, "message": "Thành công"}

def _is_filename_exists(filename_to_check: str) -> bool:
    filepath = FILENAME_FILE
    if not os.path.exists(filepath):
        return False  # Nếu file chưa tồn tại thì chắc chắn chưa có

    with open(filepath, "r") as f:
        filenames = [line.strip() for line in f]

    return filename_to_check in filenames

def insert_images_handler(images: List[FileRequest], index, filenames, embeddings):
    storedFiles: List[StoredFile] = []
    for img in images:
        img_data = _base64_to_image_bytes(img.data)
        if img_data is None or img.key is None:
            return {"message": "Đã có lỗi xảy ra"}
        else:
            storedFiles.append(StoredFile(img.key, img_data, img.metadata))
    for file in storedFiles:
        dict = insert_new_image(file.data, file.key, index, filenames, embeddings, file.metadata)
        if dict["success"] == False:
            return dict
    return {"success": True, "message": "Thành công"}

def _base64_to_image_bytes(base64_str: str) -> bytes:
    if not base64_str.strip():
        return None
    try:
        # Giải mã base64
        image_data = base64.b64decode(base64_str)
        return image_data
    except Exception as e:
        print(f"Lỗi khi chuyển base64: {e}")
        return None
    


def delete_images_handler(keys: list[str], index, filenames, embeddings) -> dict:
    not_found = [key for key in keys if key not in filenames]

    if len(not_found) == len(keys):
        return {"success": False, "message": "Không có key nào tồn tại", "not_found": not_found}

    # Xác định chỉ số các key cần xóa
    indices_to_delete = [filenames.index(key) for key in keys if key in filenames]

    # Xoá theo index (phải sort ngược để không lệch khi pop)
    for idx in sorted(indices_to_delete, reverse=True):
        filenames.pop(idx)
        embeddings = np.delete(embeddings, idx, axis=0)

    # Rebuild FAISS index
    index.reset()
    if len(embeddings) > 0:
        index.add(embeddings)

    # Lưu lại
    np.save(EMBEDDING_FILE, embeddings)
    with open(FILENAME_FILE, "w") as f:
        f.writelines([name + "\n" for name in filenames])
    faiss.write_index(index, FAISS_INDEX_FILE)

    # Xoá metadata khỏi SQLite
    delete_metadata_batch(keys)

    if not_found:
        return {"success": True, "message": f"Đã xoá {len(indices_to_delete)}/ {len(keys)} keys", "not_found": not_found}
    else:
        return {
            "success": True,
            "message": f"Đã xoá {len(indices_to_delete)}/ {len(keys)} keys",
        }
