import asyncio
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from app.core.config import get_settings
from app.utils.image_utils import load_image
router=APIRouter()
@router.post('/chat')
async def chat(request:Request,file:UploadFile=File(...),question:str|None=Form(None)):
    data=await file.read(); s=get_settings()
    if len(data)>s.max_image_size_mb*1024*1024: raise HTTPException(413,"Image exceeds configured size limit")
    return await asyncio.wait_for(asyncio.to_thread(request.app.state.pipeline.process,load_image(data),question),timeout=s.vlm_timeout_seconds)
