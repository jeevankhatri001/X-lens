import psutil
from fastapi import APIRouter
router=APIRouter()
@router.get('/metrics')
def metrics(): return {"cpu_percent":psutil.cpu_percent(),"ram_percent":psutil.virtual_memory().percent}
