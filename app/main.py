from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from app.routers import relay_router, relay_compatible_router, other_router, manager_router
from app.config import env_settings

if env_settings.open_sentry:
    import sentry_sdk

    sentry_sdk.init(
        dsn=env_settings.sentry_dsn,
        traces_sample_rate=1.0,
    )

app = FastAPI()
app.include_router(relay_router)
app.include_router(relay_compatible_router)
app.include_router(other_router)
app.include_router(manager_router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"message": exc.detail})


@app.get('/')
@app.get('/ping')
def ping():
    return "PONG"
