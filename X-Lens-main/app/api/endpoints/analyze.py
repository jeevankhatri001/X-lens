from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from app.core.config import get_settings
from app.core.dependencies import get_quality_analyzer
from app.utils.image_utils import load_image
router=APIRouter()
@router.post('/analyze')
async def analyze_image(file:UploadFile=File(...),analyzer=Depends(get_quality_analyzer)):
    data=await file.read(); s=get_settings()
    if len(data)>s.max_image_size_mb*1024*1024: raise HTTPException(413,"Image exceeds configured size limit")
    return analyzer.analyze(load_image(data)).to_dict()
