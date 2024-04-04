import argparse
import json
import pandas as pd
from tqdm import tqdm
from sentence_transformers.cross_encoder import CrossEncoder


# подсчет метрик супер моделью
def calculate_scores(model: CrossEncoder, df: pd.DataFrame) -> list:
    predictions = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        pred = model.predict([row['Оптимальный план en'], row['Предсказанный план']])
        predictions.append(round(pred))
    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", help="path to test dataset", type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset_path)
    model = CrossEncoder('taranetsdan/custom-cross_encoder', num_labels=1)

    scores = calculate_scores(model, df)
    print(scores)

    with open('results.json', 'w', encoding='utf8') as fp:
        json.dump({'scores': scores}, fp, indent=4, ensure_ascii=False)