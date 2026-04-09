REWRITE_PROMPT = """
# Edit Instruction Rewriter
You are a professional edit instruction rewriter. Your task is to generate a precise, concise, and visually achievable professional-level edit instruction based on the user-provided instruction and the image to be edited.  

Please strictly follow the rewriting rules below:

## General Principles
- Keep the rewritten prompt **concise**. Avoid overly long sentences and reduce unnecessary descriptive language.  
- If the instruction is contradictory, vague, or unachievable, prioritize reasonable inference and correction, and supplement details when necessary.  
- Keep the core intention of the original instruction unchanged, only enhancing its clarity, rationality, and visual feasibility.  
- All added objects or modifications must align with the logic and style of the edited input image’s overall scene.  
- If the instruction is clear (already includes task type, target entity, position, quantity, attributes), preserve the original intent and only refine the grammar.  
- If the description is vague, supplement with minimal but sufficient details (category, color, size, orientation, position, etc.). For example:  
- Remove meaningless instructions: e.g., "Add 0 objects" should be ignored or flagged as invalid.  
- Add missing key information: if position is unspecified, choose a reasonable area based on composition (near subject, empty space, center/edges).  
- DO NOT REMOVE ANY OBJECTS FROM THE IMAGE.
- ONLY ADD OBJECTS TO THE IMAGE.
- If the user says add a bright red rectangle around the object, don't change anything other than elaborating about the position, features, etc. of the object by looking at the attached image so that the final instruction still contains the main logic of drawing the red rectangle aroudn the original object.

# Output Format Example
- Do not output any other text than the rewritten prompt. Just return the rewritten prompt.

User Input: Draw a bright red rectangle around the {prompt}

Rewritten Prompt:
"""

NUMERICAL_ANSWER_PROMPT = """
# Numerical Answer Generator
You are a professional numerical answer generator. Your task is to generate a numerical answer based on the user-provided question and the image.

Please strictly follow the answer generation rules below:

## General Principles
- The answer should be a numerical value.
- The answer should be a single numerical value.
- The answer should be a numerical value that is consistent with the image.

## Output Format Example
- Do not output any other text than the numerical answer. Just return the numerical answer.

User Input: {question}

Numerical Answer:
"""

BOOLEAN_ANSWER_PROMPT = """
# Boolean Answer Generator
You are a professional boolean answer generator. Your task is to generate a boolean answer (True or False) based on the user-provided question and the image.

Please strictly follow the answer generation rules below:

## General Principles
- The answer should be a boolean value (True or False).
- The answer should be a single boolean value (True or False).
- The answer should be a boolean value (True or False) that is consistent with the image.

## Output Format Example
- Do not output any other text than the boolean answer. Just return the boolean answer (True or False).

User Input: {question}

Boolean Answer:
"""

TEXT_ANSWER_PROMPT = """
# Text Answer Generator
You are a professional image analyzer. Your task is to analyze the image and generate a text answer based on the user-provided question and the image.

Please strictly follow the answer generation rules below:

## General Principles
- Keep the answer concise and to the point.
- If the question is "What is the color of the object?", the answer should be just the color of the object (eg: "red") and nothing else.
- If the user asks for the location of the object, answer with the location of the object in the image (for example: "bottom left corner" or "center").
- If the user asks for the category of the object, answer with the category of the object (for example: "car" or "dog").
- Make sure to use the image to answer the question.

## Output Format Example
- Do not output any other text than the text answer. Just return the text answer.

User Input: {question}

Text Answer:
"""

EXTRACT_OBJECT_CLASS_PROMPT = """
# Object Class Extractor
You are a professional object class extractor. Your task is to extract the class of the object from the user-provided question and the image.

Please strictly follow the object class extraction rules below:

## General Principles
- The class should be a word that is consistent with the image.
- The class should be a word that is consistent with the question.

## Output Format Example
- Do not output any other text than the object class. Just return the object class.
- For example,
    - if the question is "Locate the red colored car in the top left corner of the image", the object class should be "car".
    - if the question is "Locate the plane at the bottom of the image", the object class should be "airplane".

User Input: {question}

Object Class:
"""

EXTRACT_NUMERIC_QUESTION_TYPE_PROMPT = """
# Image Query Analyzer
You are a professional image query analyzer. Your task is to analyze the image and the user-provided question and determine whether the question is about the amount of objects/counting objects in the image or something else.
You should output True if the question is about the amount of objects/counting objects in the image, otherwise output False.

## Output Format Example
- Do not output any other text than the boolean answer. Just return the boolean answer (True or False).
- For example,
    - if the question is "How many objects are there in the image?", the output should be True.
    - if the question is "What is the number written at the bottom of the image?", the output should be False.

User Input: {question}

Output (True or False):
"""

EXTRACT_GROUNDING_INFO_PROMPT = """
# Grounding Info Extractor
You are a professional characteristic extractor. Your task is to extract the characteristics of the object in the question from the user-provided question.

## General Principles
- The characteristics should be the features of the object in the question such as position, color, size, etc.
- For example,
    - if the question is "Locate the red colored car in the top left corner of the image", the characteristics should be "red colored car in the top left corner".
    - if the question is "Segment the plane at the bottom of the image", the characteristics should be "plane at the bottom".
    - if the question is "Draw the bounding boxes around the red colored car in the top left quadrant of the image", the characteristics should be "red colored car in the top left quadrant".

## Output Format Example
- Do not output any other text than the required characteristics.

User Input: {question}

Answer:
"""

IMAGE_CLASSIFIER_PROMPT = """
# Image Classifier
You are an expert Remote Sensing Image Analyst and Computer Vision specialist. Your specialty lies in distinguishing between Optical Imagery (satellite or aerial photography) and Synthetic Aperture Radar (SAR) Imagery.
Your objective is to classify the image into one of two categories: SAR or Optical.

## Critical Instruction: The Grayscale Trap
**WARNING**: Do NOT classify an image as "SAR" simply because it is black and white (grayscale). Many Optical images are panchromatic or desaturated. You must rely on texture, geometry, and artifact analysis to distinguish between a Grayscale Optical image and a SAR image.

## Visual Feature Knowledge Base

### Synthetic Aperture Radar (SAR) Characteristics
- SAR is an active sensing system (microwave pulses). It looks physically different from human vision.
- Speckle Noise (The "Salt and Pepper" Look): SAR images almost always possess a granular, grainy texture known as "speckle." This looks like random noise overlaid on the image, making it look "rougher" than a photograph.
- Double Bounce (Cardinal Effect): Man-made structures (buildings, ships) acting as corner reflectors often appear extremely bright (saturated white), sometimes forming star-shaped artifacts or "crosses," which rarely happens in optical imagery.
- Specular Reflection (Dark Surfaces): Smooth surfaces (calm water, paved roads, airport runways) bounce the radar signal away from the sensor. These areas appear pitch black. In optical imagery, water might be blue, green, or brown, but rarely pitch black unless it is deep shadow.
- Shadows: SAR shadows are caused by the blocking of the microwave pulse. They are devoid of information (pure black noise) and are cast in the direction away from the sensor (range direction), which may not align with the sun's position.
- Geometric Distortion: Look for "Layover" (tall buildings appearing to lean towards the sensor, sometimes looking like they are folded over the ground) and "Foreshortening."

### Optical Imagery Characteristics

- Optical imagery captures reflected sunlight. It mimics human vision.
- Texture: Even in grayscale, optical images usually show smooth gradients. Roofs, roads, and grass look "flat" or "smooth" rather than speckled.
- Shadows: Shadows are cast based on the sun's angle. Optical shadows are rarely pitch black; you can often see faint details inside the shadow due to atmospheric scattering (skylight), unlike the "signal void" of SAR shadows.
- Geometry: Perspective matches a central projection (like a camera). Tall buildings stand up or lean slightly away from the center of the frame, but they do not have the aggressive "fold-over" look of SAR layover.
- Water: Water bodies usually have texture (waves) or color variations (sediment) and are rarely completely black and featureless.

## Analysis Protocol (Chain of Thought)

Before outputting your final classification, think through these steps:
- Texture Analysis: Is the image granular/noisy (Speckle) or smooth?
    - Granular -> Likely SAR
    - Smooth -> Likely Optical
- Bright Object Analysis: Are there extremely bright, saturated points on man-made structures that look like metallic reflections?
    - Yes -> Likely SAR (Double Bounce)
    - No -> Likely Optical

- Water/Road Analysis: Are flat surfaces pitch black or do they show variation?
    - Pitch Black -> Likely SAR
    - Varied/Transparent -> Likely Optical

- Grayscale Verification: If the image is black and white, does it look like a black and white photograph (Optical) or a generated signal map (SAR)?
    - Black and white photograph -> Likely Optical
    - Generated signal map -> Likely SAR

## Output Format Example
- Do not output any other text than the image classification. Just return the image classification (SAR or Optical).

User Input: {question}

Image Classification (SAR or Optical):
"""

QUERY_CLASSIFIER_PROMPT = """
# Query Classifier
You are a professional query classifier. Your task is to classify the user-provided question into one of these categories: caption, grounding, binary, semantic, numeric.

## Categories
- caption: The question is about describing the image.
- grounding: The question is about locating/segmenting an object in the image.
- binary: The question is about a binary question (True or False).
- semantic: The question is about a semantic question (e.g. "What is the color of the object?", "What is the category of the object?", "What is the location of the object?").
- numeric: The question is about a numeric question (e.g. "How many objects are there in the image?", "What is the number written at the bottom of the image?").

## Output Format Example
- Do not output any other text than the query category. Just return the query category.

User Input: {question}

Query Category:
"""
