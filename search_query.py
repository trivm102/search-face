# search_query.py
import numpy as np
import face_recognition
from db import get_metadata

def search_person(query_path, index, filenames, top_k=5):
    try:
        query_image = face_recognition.load_image_file(query_path)
        query_faces = face_recognition.face_encodings(query_image)
    except Exception as e:
        print(f"❌ Lỗi khi load ảnh query: {e}")
        return {"success": False, "message": "Lỗi khi load ảnh."}

    if not query_faces:
        print("❌ Không tìm thấy khuôn mặt trong ảnh truy vấn.")
        return {"success": False, "message": "Không tìm thấy khuôn mặt trong ảnh truy vấn."}
    if index.ntotal == 0 or len(filenames) == 0:
        return {"success": False, "message": "Không tìm thấy dữ liệu"}
    if index.ntotal < top_k:
        top_k = index.ntotal
    
    query_embedding = query_faces[0].astype('float32')
    distances, indices = index.search(np.array([query_embedding]), top_k)

    results = []
    print(f"\n🔍 Kết quả tìm kiếm (Top {top_k}):")
    for i, idx in enumerate(indices[0]):
        distance = distances[0][i]
        label = "✅ Giống" if distances[0][i] < 0.4 else "⚠️ Có thể giống" if distances[0][i] < 0.6 else "❌ Khác"
        key = filenames[idx]
        metadata = get_metadata(key)
        print(f"{i+1}. {key} (Khoảng cách: {distances[0][i]:.4f}) {label}")
        results.append({
            "rank": i + 1,
            "key": key,
            "metadata": metadata,
            "distance": float(distance)
        })
    return {"success": True, "results": results}

