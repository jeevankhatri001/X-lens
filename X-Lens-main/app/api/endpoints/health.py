from fastapi import APIRouter, Request
router=APIRouter()
@router.get('/health')
def health(request:Request):
    memory=request.app.state.vlm.get_memory_stats().__dict__ if request.app.state.vlm else None
    return {"status":"ready" if request.app.state.ready else "starting","vlm_enabled":request.app.state.vlm is not None,"gpu_memory":memory}
