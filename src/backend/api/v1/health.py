from fastapi import APIRouter, Request

health_router = APIRouter(tags=["Health & Status"])

@health_router.get("/status")
async def status(request: Request):
    inference_service = request.app.state.inference_service
    config = request.app.state.config
    return {
        "status": "online",
        "lstm_ready": inference_service.lstm is not None,
        "rf_models": list(inference_service.rf.keys()),
        "hand_requirements": getattr(config, "HAND_REQUIREMENTS", {})
    }

@health_router.get("/hand_requirements")
async def hand_requirements(request: Request):
    config = request.app.state.config
    return getattr(config, "HAND_REQUIREMENTS", {})

@health_router.get("/system_words")
async def system_words(request: Request):
    inference_service = request.app.state.inference_service
    collector_service = request.app.state.collector_service
    config = request.app.state.config

    static_words = set()
    dynamic_words = set()

    # 1. Read actual trained model classes from Label Encoders
    if "static_word" in inference_service.le:
        static_words.update(inference_service.le["static_word"].classes_.tolist())
    if inference_service.lstm_le is not None:
        dynamic_words.update(inference_service.lstm_le.classes_.tolist())

    # 2. Read collected dataset classes from CSV/files if available
    stats = collector_service.get_stats()
    if "static_word" in stats and "per_label" in stats["static_word"]:
        static_words.update(stats["static_word"]["per_label"].keys())
    if "word" in stats and "per_label" in stats["word"]:
        dynamic_words.update(stats["word"]["per_label"].keys())

    # Hand requirements map
    hand_reqs = {
        "Help": 2, "School": 2, "Stop": 2, "What": 2, "Where": 2, "When": 2, "Why": 2, "Tired": 2,
        "Hello": 1, "ThankYou": 1, "Wait": 1, "Food": 1, "Sick": 1, "Sorry": 1, "Time": 1, "Toilet": 1,
        "Bye": 1, "Indian": 1, "Man": 1, "Woman": 1,
        "Bad": 1, "Call": 1, "Good": 1, "Love": 1, "Me": 1, "No": 1, "Yes": 1, "You": 1
    }
    config_reqs = getattr(config, "HAND_REQUIREMENTS", {})
    hand_reqs.update(config_reqs)

    return {
        "static_words": sorted(list(static_words)),
        "dynamic_words": sorted(list(dynamic_words)),
        "hand_requirements": hand_reqs
    }
