import asyncio
import numpy as np
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so `import storage` resolves when running the script directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import storage

async def main():
    seq = np.random.rand(5, 63).astype(np.float32)
    print("Calling save_quiz...")
    res = await storage.save_quiz(landmark_sequence=seq, inference_result={"predicted":"TEST","score":0.9,"frames_used":5}, session_id="sess1", meta={"user_id": "u1", "word_pk":123})
    print("quiz res:", json.dumps(res, ensure_ascii=False))

    print("Calling save_learning...")
    res2 = await storage.save_learning(landmark_sequence=seq, session_id="sess1", meta={"user_id":"u1","word":"안녕"})
    print("learning res:", json.dumps(res2, ensure_ascii=False))

    # List created files
    base = Path(__file__).resolve().parent.parent / 'data'
    quiz_dir = base / 'quiz'
    learning_dir = base / 'learning'

    print('\n--- data/quiz contents ---')
    if quiz_dir.exists():
        for p in sorted(quiz_dir.iterdir()):
            print(p.name)
    else:
        print('NO quiz dir')

    print('\n--- data/learning contents ---')
    if learning_dir.exists():
        for p in sorted(learning_dir.iterdir()):
            print(p.name)
    else:
        print('NO learning dir')

if __name__ == "__main__":
    asyncio.run(main())
