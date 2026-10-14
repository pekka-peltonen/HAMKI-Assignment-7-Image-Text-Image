
import argparse
import base64
import binascii
import sys
from pathlib import Path

import openai
from openai import OpenAI

VISION_MODEL = "gpt-5.6-luna"
IMAGE_MODEL = "gpt-image-2"


def main():
    parser = argparse.ArgumentParser(
        description="Image to Text to Image using OpenAI."
    )
    parser.add_argument(
        "--input", default="original.jpg"
    )
    parser.add_argument(
        "--output", default="generated.png"
    )
    args = parser.parse_args()

    image_path = Path(args.input)

    if not image_path.is_file():
        parser.error(
            f"Input image not found: {image_path}"
        )

    try:
        client = OpenAI()

        # Step 1: Read the original image
        image_bytes = image_path.read_bytes()
        image_base64 = base64.b64encode(
            image_bytes
        ).decode("ascii")

        suffix = image_path.suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        if suffix not in media_types:
            parser.error("Unsupported image format.")

        # Step 2: Analyze image using vision
        print("Analyzing original image...")

        response = client.responses.create(
            model=VISION_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Describe this image in detailed "
                                "English for an image generator. "
                                "Include the main subject, colors, "
                                "shapes, materials, surroundings, "
                                "lighting, camera angle, and style. "
                                "Do not invent details that are "
                                "not visible in the image."
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": (
                                f"data:{media_types[suffix]};base64,"
                                f"{image_base64}"
                            ),
                        },
                    ],
                }
            ],
        )

        description = response.output_text.strip()

        if not description:
            raise RuntimeError(
                "Vision model returned no description."
            )

        # Step 3: Print the description
        print("\nIMAGE DESCRIPTION:\n")
        print(description)

        # Step 4: Generate image from text only
        print("\nGenerating new image...")

        result = client.images.generate(
            model=IMAGE_MODEL,
            prompt=description,
            size="1536x1024",
            quality="medium",
            n=1,
        )

        if not result.data or not result.data[0].b64_json:
            raise RuntimeError(
                "Image generator returned no image."
            )

        generated_bytes = base64.b64decode(
            result.data[0].b64_json,
            validate=True,
        )

        # Step 5: Save generated image locally
        output_path = Path(args.output)
        output_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        output_path.write_bytes(generated_bytes)

        print(f"\nGenerated image saved: {output_path}")

    except (
        openai.OpenAIError,
        OSError,
        RuntimeError,
        binascii.Error,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
