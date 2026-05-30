from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
import threading


app = FastAPI(title="Tennis Video Analyzer API")


# =========================
# CORS 設定
# =========================
# 讓 React 前端可以呼叫 FastAPI 後端。
# 如果你前端是 Vite，預設通常是 http://localhost:5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# 專案路徑設定
# =========================
# main.py 在 project/backend/main.py
# 所以 parents[1] 會是 project/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PIPELINE_SCRIPT = PROJECT_ROOT / "video_processing_new" / "run_pipeline.py"
UPLOAD_DIR = PROJECT_ROOT / "raw_videos" / "uploads"
DATASET_DIR = PROJECT_ROOT / "dataset"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 使用者可下載的結果檔案白名單
# =========================
PUBLIC_RESULT_SPECS = {
    "overlay_video": {
        "label": "同步落點影片",
        "description": "在原始影片旁加上與影片同步的球落點",
        "patterns": [
            "overlay_bounce_landings.mp4",
        ],
    },
    "analysis_pdf": {
        "label": "分析報告 PDF",
        "description": "包含進球率、落點等等數據的分析報告",
        "patterns": [
            "analysis_report.pdf",
        ],
    },
}

# =========================
# 簡易任務狀態儲存
# =========================
# MVP 先用記憶體 dict 就好。
# 注意：如果後端重開，這裡的狀態會消失。
jobs = {}

# 避免同時跑多支影片，CPU/GPU/RAM 爆掉。
pipeline_lock = threading.Lock()

def find_first_matching_file(result_dir: Path, patterns: list[str]) -> Path | None:
    """
    根據 patterns 在 result_dir 裡找第一個符合的檔案。
    如果有多個符合，優先選擇檔案較大的，通常比較可能是正式輸出檔。
    """

    matches = []

    for pattern in patterns:
        for path in result_dir.rglob(pattern):
            if path.is_file():
                matches.append(path)

        if matches:
            # 同一組 pattern 找到就先停止，不繼續找下一組 pattern。
            break

    if not matches:
        return None

    # 排除太小的暫存檔，並優先選較大的正式輸出。
    matches.sort(key=lambda p: p.stat().st_size, reverse=True)

    return matches[0]


def get_public_result_files(job_id: str) -> list[dict]:
    """
    只回傳允許使用者下載的結果檔案。
    """

    result_dir = DATASET_DIR / job_id

    if not result_dir.exists():
        return []

    public_files = []

    for file_key, spec in PUBLIC_RESULT_SPECS.items():
        matched_file = find_first_matching_file(result_dir, spec["patterns"])

        if matched_file is None:
            continue

        relative_path = matched_file.relative_to(result_dir).as_posix()

        public_files.append(
            {
                "key": file_key,
                "label": spec["label"],
                "description": spec["description"],
                "filename": matched_file.name,
                "relative_path": relative_path,
                "size_bytes": matched_file.stat().st_size,
                "download_url": f"/api/jobs/{job_id}/download/{file_key}",
            }
        )

    return public_files

def run_pipeline_job(job_id: str, video_path: Path) -> None:
    """
    實際執行 run_pipeline.py 的背景任務。
    """

    with pipeline_lock:
        jobs[job_id]["status"] = "running"

        try:
            if not PIPELINE_SCRIPT.exists():
                raise FileNotFoundError(f"Pipeline script not found: {PIPELINE_SCRIPT}")

            cmd = [
                sys.executable,
                str(PIPELINE_SCRIPT),
                str(video_path),
            ]

            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                check=True,
            )

            jobs[job_id]["status"] = "done"
            jobs[job_id]["stdout"] = result.stdout
            jobs[job_id]["stderr"] = result.stderr

        except subprocess.CalledProcessError as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["stdout"] = e.stdout
            jobs[job_id]["stderr"] = e.stderr
            jobs[job_id]["error"] = str(e)

        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)


@app.get("/")
def root():
    return {
        "message": "Tennis Video Analyzer API is running.",
        "docs": "/docs",
    }


@app.post("/api/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    上傳影片，建立 job，背景執行 run_pipeline.py。
    """

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    allowed_extensions = {".mp4", ".mov", ".avi", ".mkv"}
    original_suffix = Path(file.filename).suffix.lower()

    if original_suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only .mp4, .mov, .avi, .mkv files are allowed.",
        )

    # 用 uuid 當 job_id，避免不同使用者上傳同名影片互相覆蓋。
    job_id = str(uuid.uuid4())

    # 關鍵：影片檔名用 job_id。
    # 因為 run_pipeline.py 會用 video_path.stem 當輸出資料夾名稱。
    # 例如：abc123.mp4 -> dataset/abc123/
    saved_video_path = UPLOAD_DIR / f"{job_id}{original_suffix}"

    try:
        with saved_video_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()

    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "original_filename": file.filename,
        "saved_video_path": str(saved_video_path),
    }

    background_tasks.add_task(run_pipeline_job, job_id, saved_video_path)

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Video uploaded. Analysis has been queued.",
    }


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    """
    查詢 job 狀態。
    """

    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = jobs[job_id]

    response = {
        "job_id": job_id,
        "status": job["status"],
        "original_filename": job.get("original_filename"),
    }

    if job["status"] == "failed":
        response["error"] = job.get("error")
        response["stderr"] = job.get("stderr")

    return response


@app.get("/api/jobs/{job_id}/files")
def list_public_result_files(job_id: str):
    """
    只列出使用者可下載的成果檔案。
    不暴露 dataset/{job_id}/ 裡面的所有中間檔。
    """

    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = jobs[job_id]

    if job.get("status") != "done":
        return {
            "job_id": job_id,
            "status": job.get("status"),
            "message": "Analysis is not finished yet.",
            "files": [],
        }

    result_dir = DATASET_DIR / job_id

    if not result_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Result folder not found: {result_dir}",
        )

    public_files = get_public_result_files(job_id)

    return {
        "job_id": job_id,
        "status": job.get("status"),
        "files": public_files,
    }


@app.get("/api/jobs/{job_id}/download/{file_key}")
def download_public_result_file(job_id: str, file_key: str):
    """
    根據 file_key 下載指定成果檔案。
    例如：
    - overlay_video
    - analysis_pdf
    """

    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = jobs[job_id]

    if job.get("status") != "done":
        raise HTTPException(
            status_code=400,
            detail="Analysis is not finished yet.",
        )

    if file_key not in PUBLIC_RESULT_SPECS:
        raise HTTPException(
            status_code=404,
            detail="This file is not available for download.",
        )

    result_dir = DATASET_DIR / job_id

    if not result_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Result folder not found: {result_dir}",
        )

    spec = PUBLIC_RESULT_SPECS[file_key]
    matched_file = find_first_matching_file(result_dir, spec["patterns"])

    if matched_file is None:
        raise HTTPException(
            status_code=404,
            detail=f"{file_key} was not generated.",
        )

    return FileResponse(
        path=matched_file,
        filename=matched_file.name,
        media_type="application/octet-stream",
    )