import json
import pandas as pd
import numpy as np


def json_to_df(json_data: str|dict,
        feature_set: list((str, int))) -> pd.DataFrame:
    '''
    Преобразование json'а указанного вида в датафрейм.
    @param json_data:
        : json вида (как словарь или строка):
            {
                "features": [
                    {
                        "Element_ID": "627604",
                        "Type": "ДБ-одностворчатый-металлический-глухой-внутренний-2100х1300мм_правый т.2",
                        "Family": "ДБ-1ый-метал",
                    }
                ],
                "feature_payload": {
                    "Type": {
                        "property_domain_name": "KSI",
                        "property_code": "XNT_0002",
                        "name": "Тип",
                        "data_type": "String",
                        "sort_value": 1
                    },
                    "Family": {
                        "property_domain_name": "KSI",
                        "property_code": "XNT_0004",
                        "name": "Категория",
                        "data_type": "String",
                        "sort_value": 2
                    }
                }
            }
    @param feature_set:
        : список кортежей, представляющих требуемый набор параметров и их порядок;
        : например: [('Element_ID', 0), ('XNT_0002', 1), ('XNT_0004', 2), ('XNT_0004', 3)], строка должна
            представлять код, а число - значение sort_value целевого признака;
        : признаки, которых нет в джейсоне, будут добавлены как солбцы, и будут содержать np.nan;
        : лишние признаки из джейсона не будут добавлены в датафрейм;

    @return -> pd.DataFrame:
        : датафрейм после преобразования джейсона;
        : значения вида пустой строки будут заменены на np.nan;
        : например:
            00_Element_ID | 01_XNT_0002 | 02_XNT_0004 | 03_XNT_0004

            627604        | ДБ-од...    | ДБ-1ы...    | Nan     
    '''
    # Извлечение информации из json'а
    if isinstance(json_data, str):
        json_data = json.loads(json_data)
    mapping = json_data['feature_payload']
    df = pd.DataFrame(json_data['features'])

    order = []
    to_drop = []
    # Определение новых имен в соответсвии с маппированием
    for col in df.columns:
        if col != 'Element_ID':
            code = mapping[col]['property_code']
            sort_val = mapping[col]['sort_value']
            if col in mapping and (code, sort_val) in feature_set:
                order.append((code, sort_val))
            else:
                to_drop.append(col)
        else:
            # 'Element_ID' рассматривается отдельно, как не требующий маппирования
            if (col, 0) in feature_set:
                order.append((col, 0))
            else:
                to_drop.append(col)

    # Удаление не нужных (не заданных в feature_set) признаков
    df.drop(columns=to_drop, inplace=True)

    df.columns = [('0' + str(x[1]))[-2:] + '_' + x[0] for x in order]

    # Дополнение недостающими столбцами и сортировка признаков
    for col,sort_val in feature_set:
        if (col, sort_val) not in order:
            order.append((col, sort_val))
            df[('0' + str(sort_val))[-2:] + '_' + col] = np.nan

    df.sort_index(axis=1, inplace=True)
    
    return df.replace(r'^\s*$', np.nan, regex=True)