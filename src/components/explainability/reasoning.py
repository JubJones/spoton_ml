import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def generate_reasoning_text(
    detection_result: Dict[str, Any],
    explanation_type: str = "Grad-CAM Focus",
    has_visualization: bool = False
) -> str:
    """
    Generates a human-readable explanation for a single detection prediction.

    Args:
        detection_result: A dictionary containing at least 'label', 'score'.
                          Should ideally contain 'box' as well.
                          Example: {'box': [x1,y1,x2,y2], 'label': 1, 'score': 0.95}
        explanation_type: The type of explanation method used (e.g., "Grad-CAM Focus",
                          "Confidence Score").
        has_visualization: Boolean indicating if a visual explanation (heatmap) is available.

    Returns:
        A string containing the reasoning text.
    """
    label = detection_result.get("label", "Unknown")
    score = detection_result.get("score", 0.0)
    box = detection_result.get("box", None)

    # --- Basic Reasoning based on score and class ---
    reasoning = f"Detected object classified as 'person' (label {label}) with a confidence score of {score:.2f}. "

    if score >= 0.9:
        reasoning += "This high confidence suggests the model found strong visual evidence matching features learned for the 'person' class. "
    elif score >= 0.7:
        reasoning += "This moderate confidence indicates the model found recognizable features, but potentially with some ambiguity or minor occlusion. "
    else:
        reasoning += "This lower confidence suggests the model found some indicative features, but there might be significant occlusion, unusual pose, distance, or conflicting background elements. "

    # --- Add context if XAI visualization is available ---
    if has_visualization:
        reasoning += f"The associated visualization ({explanation_type}) highlights the specific image regions that most influenced this classification decision. "
        if box:
             reasoning += f"These influential regions are located primarily within the detected bounding box [{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]. "
        reasoning += "Reviewing the heatmap overlay can provide further insight into which parts of the object (e.g., head, torso) contributed most to the prediction."
    else:
        reasoning += "No specific feature attribution visualization was generated for this prediction."

    logger.debug(f"Generated reasoning for label {label}, score {score:.2f}: {reasoning}")
    return reasoning