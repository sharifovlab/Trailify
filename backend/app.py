from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import gpxpy
from geopy.distance import geodesic
from io import BytesIO
import json

app = FastAPI(title="Trail Difficulty API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://trailify-ai.vercel.app",
        "http://localhost:5173",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
try:
    model_data = joblib.load('trail_model_v1.0.0.pkl')
    model = model_data['model']
    scaler = model_data['scaler']
    feature_columns = model_data['feature_columns']
    print("✅ Модель загружена успешно")
    print(f"📊 Количество фичей: {len(feature_columns)}")
    print(f"📋 Список фичей: {feature_columns}")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    model = None


class GPXAnalyzer:
    """Анализирует GPX файлы"""
    
    def __init__(self, gpx_content):
        self.gpx = gpxpy.parse(gpx_content)
        self.points = []
        self.extract_points()
    
    def extract_points(self):
        for track in self.gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    if point.elevation:
                        self.points.append({
                            'lat': point.latitude,
                            'lon': point.longitude,
                            'ele': point.elevation
                        })
    
    def calculate_features(self):
        if len(self.points) < 2:
            raise ValueError("Недостаточно GPS точек в файле")
        
        print(f"\n🔍 АНАЛИЗ ТРЕЙЛА НАЧАТ")
        print(f"📍 Количество GPS точек: {len(self.points)}")
        
        # Distance calculation
        total_distance = 0
        elevations = [p['ele'] for p in self.points]
        
        for i in range(len(self.points) - 1):
            p1 = (self.points[i]['lat'], self.points[i]['lon'])
            p2 = (self.points[i+1]['lat'], self.points[i+1]['lon'])
            total_distance += geodesic(p1, p2).meters
        
        print(f"📏 Общая дистанция: {total_distance/1000:.2f} км")
        
        # Sinuosity
        start = (self.points[0]['lat'], self.points[0]['lon'])
        end = (self.points[-1]['lat'], self.points[-1]['lon'])
        straight_distance = geodesic(start, end).meters
        sinuosity = total_distance / straight_distance if straight_distance > 0 else 1.0
        
        print(f"🔄 Извилистость (sinuosity): {sinuosity:.2f}")
        
        # Elevation metrics
        min_ele = min(elevations)
        max_ele = max(elevations)
        elevation_range = max_ele - min_ele
        mean_ele = np.mean(elevations)
        
        print(f"⛰️  Мин высота: {min_ele:.1f}м, Макс: {max_ele:.1f}м, Диапазон: {elevation_range:.1f}м")
        
        # Elevation gain/loss
        elevation_gain = 0
        elevation_loss = 0
        grades = []
        
        for i in range(len(self.points) - 1):
            ele_diff = self.points[i+1]['ele'] - self.points[i]['ele']
            p1 = (self.points[i]['lat'], self.points[i]['lon'])
            p2 = (self.points[i+1]['lat'], self.points[i+1]['lon'])
            dist = geodesic(p1, p2).meters
            
            if ele_diff > 0:
                elevation_gain += ele_diff
            else:
                elevation_loss += abs(ele_diff)
            
            if dist > 0:
                grade = (ele_diff / dist) * 100
                grades.append(grade)
        
        print(f"📈 Набор высоты: {elevation_gain:.0f}м, Потеря: {elevation_loss:.0f}м")
        
        # Grade statistics
        max_grade = max(grades) if grades else 0
        min_grade = min(grades) if grades else 0
        steep_gt_10 = sum(1 for g in grades if abs(g) > 10)
        steep_gt_15 = sum(1 for g in grades if abs(g) > 15)
        steep_uphill = sum(1 for g in grades if g > 10)
        steep_downhill = sum(1 for g in grades if g < -10)
        
        pct_steep_gt_10 = (steep_gt_10 / len(grades) * 100) if grades else 0
        pct_steep_gt_15 = (steep_gt_15 / len(grades) * 100) if grades else 0
        pct_steep_uphill = (steep_uphill / len(grades) * 100) if grades else 0
        pct_steep_downhill = (steep_downhill / len(grades) * 100) if grades else 0
        
        print(f"📊 Макс уклон: {max_grade:.1f}%, Мин: {min_grade:.1f}%")
        print(f"⚠️  Крутых сегментов >10%: {pct_steep_gt_10:.1f}%")
        
        # Turn analysis
        turn_count = 0
        turn_angles = []
        
        for i in range(1, len(self.points) - 1):
            lat_diff1 = self.points[i]['lat'] - self.points[i-1]['lat']
            lon_diff1 = self.points[i]['lon'] - self.points[i-1]['lon']
            lat_diff2 = self.points[i+1]['lat'] - self.points[i]['lat']
            lon_diff2 = self.points[i+1]['lon'] - self.points[i]['lon']
            
            angle = abs(np.arctan2(lat_diff2, lon_diff2) - np.arctan2(lat_diff1, lon_diff1))
            angle = np.degrees(angle)
            
            if angle > 30:
                turn_count += 1
                turn_angles.append(angle)
        
        turn_mean = np.mean(turn_angles) if turn_angles else 0
        turn_std = np.std(turn_angles) if turn_angles else 0
        turn_max = max(turn_angles) if turn_angles else 0
        num_sharp_turns = sum(1 for a in turn_angles if a > 90)
        pct_sharp_turns = (num_sharp_turns / len(turn_angles) * 100) if turn_angles else 0
        
        print(f"🔀 Поворотов: {turn_count}, Резких (>90°): {num_sharp_turns}")
        
        distance_km = total_distance / 1000
        climb_rate = (elevation_gain / distance_km) if distance_km > 0 else 0
        descent_rate = (elevation_loss / distance_km) if distance_km > 0 else 0
        turns_per_km = (turn_count / distance_km) if distance_km > 0 else 0
        
        print(f"📈 Скорость набора высоты: {climb_rate:.1f} м/км")
        
        # Create elevation profile
        elevation_profile = []
        elevation_profile.append({'distance': 0, 'elevation': round(self.points[0]['ele'], 1)})
        
        distances = [0]
        for i in range(1, len(self.points)):
            p1 = (self.points[i-1]['lat'], self.points[i-1]['lon'])
            p2 = (self.points[i]['lat'], self.points[i]['lon'])
            distances.append(distances[-1] + geodesic(p1, p2).meters / 1000)
        
        total_dist_km = distances[-1]
        sample_distance_km = max(0.2, total_dist_km / 300)
        
        next_sample_dist = sample_distance_km
        for i in range(1, len(self.points)):
            if distances[i] >= next_sample_dist or i == len(self.points) - 1:
                elevation_profile.append({
                    'distance': round(distances[i], 2),
                    'elevation': round(self.points[i]['ele'], 1)
                })
                next_sample_dist += sample_distance_km
        
        print(f"📊 Точек в профиле высот: {len(elevation_profile)}")
        
        # Trail path for map
        trail_path = []
        sample_interval_map = max(1, len(self.points) // 500)
        for i in range(0, len(self.points), sample_interval_map):
            trail_path.append([self.points[i]['lat'], self.points[i]['lon']])
        
        # Return all features
        features = {
            'num_points': len(self.points),
            'total_distance_m': total_distance,
            'sinuosity': sinuosity,
            'min_elevation_m': min_ele,
            'max_elevation_m': max_ele,
            'elevation_range_m': elevation_range,
            'mean_elevation_m': mean_ele,
            'total_elevation_gain_m': elevation_gain,
            'total_elevation_loss_m': elevation_loss,
            'climb_rate_m_per_km': climb_rate,
            'descent_rate_m_per_km': descent_rate,
            'max_grade_pct': max_grade,
            'min_grade_pct': min_grade,
            'pct_segments_steep_gt_10': pct_steep_gt_10,
            'pct_segments_steep_gt_15': pct_steep_gt_15,
            'pct_steep_uphill': pct_steep_uphill,
            'pct_steep_downhill': pct_steep_downhill,
            'turn_count': turn_count,
            'turn_mean_deg': turn_mean,
            'turn_std_deg': turn_std,
            'turn_max_deg': turn_max,
            'num_sharp_turns': num_sharp_turns,
            'pct_sharp_turns': pct_sharp_turns,
            'turns_deg_per_km': turns_per_km,
            'surface_unpaved_pct': 50.0,
            'surface_dirt_pct': 30.0,
            'surface_ground_pct': 20.0,
            'surface_ground_other_pct': 0.0,
            'elevation_profile': elevation_profile,
            'trail_path': trail_path
        }
        
        return features


def get_difficulty_class(score):
    if score < 35:
        return 'Easy'
    elif score < 55:
        return 'Moderate'
    elif score < 75:
        return 'Hard'
    elif score < 90:
        return 'Very Hard'
    else:
        return 'Extreme'


@app.get("/")
async def root():
    return {
        "message": "Trail Difficulty API",
        "status": "online",
        "model_loaded": model is not None
    }


@app.post("/predict")
async def predict_difficulty(file: UploadFile = File(...),surface_type: str = "mixed"):
    if not file.filename.endswith('.gpx'):
        raise HTTPException(status_code=400, detail="Только GPX файлы поддерживаются")
    
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")
    
    try:
        # Read and parse GPX
        contents = await file.read()
        gpx_content = contents.decode('utf-8')
        
        # Analyze trail
        analyzer = GPXAnalyzer(gpx_content)
        features = analyzer.calculate_features()
        
        # Prepare for prediction
        df = pd.DataFrame([features])
        
        # Add derived features
        df['distance_km'] = df['total_distance_m'] / 1000
        df['elevation_per_km'] = df['total_elevation_gain_m'] / df['distance_km']
        df['avg_grade'] = (df['max_grade_pct'] + abs(df['min_grade_pct'])) / 2
        df['grade_range'] = df['max_grade_pct'] - df['min_grade_pct']
        df['steep_ratio'] = df['pct_segments_steep_gt_15'] / (df['pct_segments_steep_gt_10'] + 1)
        df['technical_index'] = df['pct_sharp_turns'] * df['turn_max_deg'] / 100
        df['surface_difficulty'] = (
            df['surface_unpaved_pct'] * 0.5 + 
            df['surface_dirt_pct'] * 0.3 + 
            df['surface_ground_pct'] * 0.2
        )
        
        print(f"\n" + "="*60)
        print(f"🤖 ПРЕДСКАЗАНИЕ МОДЕЛИ")
        print(f"="*60)
        
        # Check which features are available
        print(f"\n📋 ПРОВЕРКА ФИЧЕЙ:")
        missing_features = []
        for col in feature_columns:
            if col in df.columns:
                print(f"  ✅ {col}: {df[col].values[0]:.3f}")
            else:
                print(f"  ❌ {col}: ОТСУТСТВУЕТ!")
                missing_features.append(col)
        
        if missing_features:
            print(f"\n⚠️  ВНИМАНИЕ: Отсутствует {len(missing_features)} фичей!")
            print(f"Отсутствующие фичи: {missing_features}")
        
        # Predict
        X = df[feature_columns]
        
        print(f"\n📊 ЗНАЧЕНИЯ ФИЧЕЙ ДЛЯ МОДЕЛИ (первые 10):")
        for i, col in enumerate(feature_columns[:10]):
            if col in X.columns:
                print(f"  {i+1}. {col}: {X[col].values[0]:.3f}")
        
        X_scaled = scaler.transform(X)
        
        print(f"\n🔢 НОРМАЛИЗОВАННЫЕ ЗНАЧЕНИЯ (первые 10):")
        for i in range(min(10, len(feature_columns))):
            print(f"  {i+1}. {feature_columns[i]}: {X_scaled[0][i]:.3f}")
        
        difficulty_score = float(model.predict(X_scaled)[0])
        
        print(f"\n🎯 СЫРОЙ РЕЗУЛЬТАТ МОДЕЛИ: {difficulty_score:.2f}")
        
        # Clamp score
        difficulty_score = max(0, min(100, difficulty_score))
        
        print(f"✅ ФИНАЛЬНЫЙ БАЛЛ: {difficulty_score:.2f}")
        
        difficulty_class = get_difficulty_class(difficulty_score)
        
        print(f"🏆 КАТЕГОРИЯ: {difficulty_class}")
        print(f"="*60 + "\n")
        
        # Return results
        return {
            'difficulty_score': round(difficulty_score, 1),
            'difficulty_class': difficulty_class,
            'distance_km': round(df['distance_km'].values[0], 2),
            'elevation_gain': round(features['total_elevation_gain_m'], 0),
            'elevation_loss': round(features['total_elevation_loss_m'], 0),
            'max_elevation': round(features['max_elevation_m'], 0),
            'elevation_range': round(features['elevation_range_m'], 0),
            'climb_rate': round(features['climb_rate_m_per_km'], 1),
            'max_grade': round(features['max_grade_pct'], 1),
            'steep_segments': round(features['pct_segments_steep_gt_10'], 1),
            'sinuosity': round(features['sinuosity'], 2),
            'elevation_profile': features['elevation_profile'],
            'trail_path': features['trail_path']
        }
        
    except Exception as e:
        print(f"❌ ОШИБКА: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
