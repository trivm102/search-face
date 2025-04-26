# main.py
from fastapi import FastAPI, File, UploadFile, Body
from search_query import search_person
from image_handler import insert_images_handler, delete_images_handler
import io
import numpy as np
import faiss
import os
from typing import List
from contextlib import asynccontextmanager
from constant import EMBEDDING_FILE, FAISS_INDEX_FILE, FILENAME_FILE, DATA_FOLDER
from request_model.file_request import FileRequest
from request_model.keys_request import KeysRequest
from db import init_metadata_db

# embeddings = None
# filenames = None
# index = None

def initialize():
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(f"Thư mục '{DATA_FOLDER}' đã được tạo.")
    # 1. embeddings.npy
    if not os.path.exists(EMBEDDING_FILE):
        print(f"🆕 Tạo {EMBEDDING_FILE} rỗng...")
        empty_embeddings = np.empty((0, 128), dtype='float32')
        np.save(EMBEDDING_FILE, empty_embeddings)

    # 2. filenames.txt
    if not os.path.exists(FILENAME_FILE):
        print(f"🆕 Tạo {FILENAME_FILE} rỗng...")
        with open(FILENAME_FILE, "w") as f:
            pass

    # 3. index.faiss
    if not os.path.exists(FAISS_INDEX_FILE):
        print(f"🆕 Tạo {FAISS_INDEX_FILE} rỗng...")
        indexFile = faiss.IndexFlatL2(128)  # FAISS dùng vector 128 chiều
        faiss.write_index(indexFile, FAISS_INDEX_FILE)
    
    #4. database
    init_metadata_db()

    print("✅ Đã kiểm tra và tạo các file cần thiết nếu chưa tồn tại.")
    # global embeddings, filenames, index
    # embeddings = np.load(EMBEDDING_FILE)
    # with open(FILENAME_FILE, "r") as f:
    #     filenames = [line.strip() for line in f]
    # index = faiss.read_index(FAISS_INDEX_FILE)

# ⚙️ Lifespan event
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 FastAPI khởi động...")
    initialize()
    print("📂 Loading dữ liệu...")
    yield

app = FastAPI(lifespan=lifespan)


@app.post("/search")
async def search_face(file: UploadFile = File(...)):
    _, filenames, index = _load_data()
    image_bytes = await file.read()
    result = search_person(io.BytesIO(image_bytes), index, filenames)
    return result

@app.post("/insert-images")
async def insert_images(images: List[FileRequest]):
    embeddings, filenames, index = _load_data()
    result = insert_images_handler(images, index, filenames, embeddings)
    return result

@app.delete("/delete-images")
async def delete_images(request: KeysRequest= Body(...)):
    embeddings, filenames, index = _load_data()
    keys = request.keys
    result = delete_images_handler(keys, index, filenames, embeddings)
    return result

def _load_data():
    embeddings = np.load(EMBEDDING_FILE)
    with open(FILENAME_FILE, "r") as f:
        filenames = [line.strip() for line in f]
    index = faiss.read_index(FAISS_INDEX_FILE)
    return embeddings, filenames, index
