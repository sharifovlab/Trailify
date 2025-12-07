import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { Mountain, TrendingUp, Gauge, MapPin, Upload, CheckCircle2, XCircle, TrendingDown, Maximize2, Navigation, AlertCircle } from 'lucide-react';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function TrailPredictor() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [surfaceType, setSurfaceType] = useState('mixed');

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = async (uploadedFile) => {
    // Validate file type
    if (!uploadedFile.name.endsWith('.gpx')) {
      setError('Please upload a GPX file (.gpx extension)');
      return;
    }

    // Validate file size (10MB max)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (uploadedFile.size > maxSize) {
      setError(`File too large: ${(uploadedFile.size / (1024 * 1024)).toFixed(1)}MB (max 10MB)`);
      return;
    }

    setFile(uploadedFile);
    setError(null);
    setLoading(true);

    // Call backend API
    try {
      const formData = new FormData();
      formData.append('file', uploadedFile);
      formData.append('surface_type', surfaceType);
      
      const response = await fetch(`${API_URL}/predict?surface_type=${surfaceType}`, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Validate response
      if (!data.difficulty_score || data.difficulty_score < 0 || data.difficulty_score > 100) {
        throw new Error('Invalid prediction received from server');
      }
      
      setPrediction(data);
      setLoading(false);
    } catch (err) {
      console.error('API Error:', err);
      setError(err.message || 'Failed to analyze trail. Please try again.');
      setLoading(false);
    }
  };

  const getDifficultyColor = (score) => {
    if (score < 35) return { bg: '#dcfce7', text: '#166534', name: 'Easy', desc: 'Suitable for beginners' };
    if (score < 55) return { bg: '#fef3c7', text: '#92400e', name: 'Moderate', desc: 'Some experience recommended' };
    if (score < 75) return { bg: '#fed7aa', text: '#9a3412', name: 'Hard', desc: 'Experienced hikers' };
    if (score < 90) return { bg: '#fecaca', text: '#991b1b', name: 'Very Hard', desc: 'Very experienced only' };
    return { bg: '#ddd6fe', text: '#5b21b6', name: 'Extreme', desc: 'Expert level required' };
  };

  const surfaceOptions = [
    { value: 'paved', label: 'Paved', emoji: '🛣️' },
    { value: 'gravel', label: 'Gravel', emoji: '🪨' },
    { value: 'dirt', label: 'Dirt', emoji: '🟤' },
    { value: 'technical', label: 'Technical/Rocky', emoji: '⛰️' },
    { value: 'mixed', label: 'Mixed', emoji: '🌍' }
  ];

  return (
    <div style={{
      minHeight: '100vh',
      background: 'white',
      padding: '2rem',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
        {/* Header */}
        <div style={{
          textAlign: 'center',
          marginBottom: '3rem',
          padding: '2rem',
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          borderRadius: '20px',
          color: 'white'
        }}>
          <div style={{
            fontSize: '3rem',
            marginBottom: '0.5rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '1rem'
          }}>
            <Mountain size={48} />
            <h1 style={{ margin: 0, fontWeight: 700 }}>Trailify</h1>
          </div>
          <p style={{
            fontSize: '1.1rem',
            opacity: 0.95,
            margin: 0,
            fontWeight: 400
          }}>
            AI-Powered Trail Difficulty Analysis
          </p>
        </div>

        {/* Upload Area */}
        {!prediction && (
          <div>
            {/* Surface Type Selection */}
            <div style={{
              background: 'white',
              borderRadius: '16px',
              padding: '1.5rem',
              marginBottom: '1.5rem',
              boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
              border: '1px solid #e5e7eb'
            }}>
              <label style={{
                display: 'block',
                fontSize: '0.875rem',
                fontWeight: 600,
                color: '#374151',
                marginBottom: '0.75rem',
                textTransform: 'uppercase',
                letterSpacing: '0.5px'
              }}>
                Trail Surface Type
              </label>
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(5, 1fr)',
                gap: '0.75rem'
              }}>
                {surfaceOptions.map(option => (
                  <button
                    key={option.value}
                    onClick={() => setSurfaceType(option.value)}
                    style={{
                      padding: '1rem',
                      background: surfaceType === option.value ? '#667eea' : '#f9fafb',
                      color: surfaceType === option.value ? 'white' : '#374151',
                      border: surfaceType === option.value ? '2px solid #667eea' : '2px solid #e5e7eb',
                      borderRadius: '12px',
                      cursor: 'pointer',
                      fontSize: '0.875rem',
                      fontWeight: 600,
                      transition: 'all 0.2s',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      gap: '0.5rem'
                    }}
                    onMouseEnter={(e) => {
                      if (surfaceType !== option.value) {
                        e.currentTarget.style.background = '#f3f4f6';
                        e.currentTarget.style.borderColor = '#d1d5db';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (surfaceType !== option.value) {
                        e.currentTarget.style.background = '#f9fafb';
                        e.currentTarget.style.borderColor = '#e5e7eb';
                      }
                    }}
                  >
                    <span style={{ fontSize: '1.5rem' }}>{option.emoji}</span>
                    <span>{option.label}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* File Upload */}
            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => !loading && document.getElementById('fileInput').click()}
              style={{
                background: 'white',
                borderRadius: '20px',
                padding: '4rem 2rem',
                textAlign: 'center',
                cursor: loading ? 'not-allowed' : 'pointer',
                border: dragActive ? '3px dashed #667eea' : '3px dashed #e5e7eb',
                transition: 'all 0.3s ease',
                marginBottom: '2rem',
                boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
                opacity: loading ? 0.6 : 1
              }}
            >
              <input
                id="fileInput"
                type="file"
                accept=".gpx"
                onChange={handleChange}
                disabled={loading}
                style={{ display: 'none' }}
              />
              
              {loading ? (
                <div>
                  <div style={{
                    width: '60px',
                    height: '60px',
                    border: '4px solid #f3f4f6',
                    borderTop: '4px solid #667eea',
                    borderRadius: '50%',
                    margin: '0 auto 1.5rem',
                    animation: 'spin 1s linear infinite'
                  }}>
                    <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
                  </div>
                  <p style={{ fontSize: '1.2rem', fontWeight: 600, color: '#374151', margin: 0 }}>
                    Analyzing your trail...
                  </p>
                  <p style={{ fontSize: '0.875rem', color: '#6b7280', marginTop: '0.5rem' }}>
                    This may take a few seconds
                  </p>
                </div>
              ) : (
                <>
                  <Upload size={64} color="#667eea" strokeWidth={2} style={{ margin: '0 auto 1.5rem' }} />
                  <h3 style={{
                    fontSize: '1.5rem',
                    fontWeight: 600,
                    color: '#1f2937',
                    marginBottom: '0.75rem'
                  }}>
                    Drop your GPX file here
                  </h3>
                  <p style={{
                    fontSize: '1rem',
                    color: '#6b7280',
                    margin: 0
                  }}>
                    or click to browse • Supports .gpx files from Strava, Garmin, AllTrails
                  </p>
                  <p style={{
                    fontSize: '0.875rem',
                    color: '#9ca3af',
                    marginTop: '0.5rem'
                  }}>
                    Maximum file size: 10MB
                  </p>
                  {error && (
                    <div style={{
                      marginTop: '1.5rem',
                      padding: '1rem',
                      background: '#fee2e2',
                      color: '#991b1b',
                      borderRadius: '10px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '0.5rem'
                    }}>
                      <AlertCircle size={20} />
                      {error}
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        )}

        {/* Results */}
        {prediction && (
          <div style={{ animation: 'fadeIn 0.5s ease' }}>
            <style>{`@keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }`}</style>
            
            {/* File Info */}
            <div style={{
              background: 'white',
              borderRadius: '16px',
              padding: '1.5rem',
              marginBottom: '1.5rem',
              display: 'flex',
              alignItems: 'center',
              gap: '1rem',
              boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
              border: '1px solid #e5e7eb'
            }}>
              <CheckCircle2 size={32} color="#10b981" />
              <div style={{ flex: 1 }}>
                <p style={{ margin: 0, fontWeight: 600, fontSize: '1.1rem', color: '#1f2937' }}>
                  {file?.name || 'Trail Analysis Complete'}
                </p>
                <p style={{ fontSize: '0.875rem', color: '#6b7280', marginTop: '0.25rem', margin: 0 }}>
                  {prediction.num_points} GPS points • Surface: {prediction.surface_type}
                </p>
              </div>
              <button
                onClick={() => { setPrediction(null); setFile(null); setError(null); }}
                style={{
                  padding: '0.5rem 1.5rem',
                  background: '#f3f4f6',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  fontWeight: 500,
                  color: '#374151',
                  transition: 'background 0.2s'
                }}
                onMouseEnter={(e) => e.currentTarget.style.background = '#e5e7eb'}
                onMouseLeave={(e) => e.currentTarget.style.background = '#f3f4f6'}
              >
                New Analysis
              </button>
            </div>

            {/* Difficulty Score - Hero Card */}
            <div style={{
              background: 'white',
              borderRadius: '20px',
              padding: '2.5rem',
              marginBottom: '1.5rem',
              textAlign: 'center',
              boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
              border: '1px solid #e5e7eb',
              position: 'relative',
              overflow: 'hidden'
            }}>
              <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                height: '6px',
                background: `linear-gradient(90deg, ${getDifficultyColor(prediction.difficulty_score).bg}, ${getDifficultyColor(prediction.difficulty_score).text})`
              }} />
              
              <p style={{
                fontSize: '0.875rem',
                fontWeight: 600,
                color: '#6b7280',
                textTransform: 'uppercase',
                letterSpacing: '1px',
                margin: '0 0 1rem 0'
              }}>
                Trail Difficulty Score
              </p>
              
              <div style={{
                fontSize: '5rem',
                fontWeight: 800,
                background: `linear-gradient(135deg, ${getDifficultyColor(prediction.difficulty_score).text}, #667eea)`,
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                marginBottom: '1rem',
                lineHeight: 1
              }}>
                {Math.round(prediction.difficulty_score)}
              </div>
              
              <div style={{
                display: 'inline-block',
                padding: '0.75rem 2rem',
                background: getDifficultyColor(prediction.difficulty_score).bg,
                color: getDifficultyColor(prediction.difficulty_score).text,
                borderRadius: '12px',
                fontSize: '1.25rem',
                fontWeight: 700,
                marginBottom: '0.5rem'
              }}>
                {getDifficultyColor(prediction.difficulty_score).name}
              </div>
              
              <p style={{
                fontSize: '0.875rem',
                color: '#6b7280',
                margin: 0
              }}>
                {getDifficultyColor(prediction.difficulty_score).desc}
              </p>
            </div>

            {/* Trail Metrics Grid */}
            <div style={{
              background: 'white',
              borderRadius: '20px',
              padding: '2rem',
              marginBottom: '1.5rem',
              boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
              border: '1px solid #e5e7eb'
            }}>
              <h3 style={{
                fontSize: '1.25rem',
                fontWeight: 600,
                color: '#1f2937',
                marginBottom: '1.5rem',
                margin: '0 0 1.5rem 0'
              }}>
                Trail Metrics
              </h3>
              
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: '1rem'
              }}>
                <MetricCard icon={MapPin} label="Distance" value={`${prediction.distance_km} km`} color="#3b82f6" />
                <MetricCard icon={Mountain} label="Max Elevation" value={`${Math.round(prediction.max_elevation)} m`} color="#8b5cf6" />
                <MetricCard icon={TrendingUp} label="Total Gain" value={`${Math.round(prediction.elevation_gain)} m`} color="#10b981" />
                <MetricCard icon={TrendingDown} label="Total Loss" value={`${Math.round(prediction.elevation_loss)} m`} color="#ef4444" />
                <MetricCard icon={Gauge} label="Climb Rate" value={`${prediction.climb_rate} m/km`} color="#f59e0b" />
                <MetricCard icon={TrendingUp} label="Max Grade" value={`${prediction.max_grade}%`} color="#ec4899" />
                <MetricCard icon={Maximize2} label="Elev. Range" value={`${Math.round(prediction.elevation_range)} m`} color="#06b6d4" />
                <MetricCard icon={Navigation} label="Sinuosity" value={prediction.sinuosity} color="#14b8a6" />
                <MetricCard icon={Mountain} label="Steep Segments" value={`${prediction.steep_segments}%`} color="#f97316" />
              </div>
            </div>

            {/* Elevation Profile */}
            <div style={{
              background: 'white',
              borderRadius: '20px',
              padding: '2rem',
              marginBottom: '1.5rem',
              boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
              border: '1px solid #e5e7eb'
            }}>
              <h3 style={{
                fontSize: '1.25rem',
                fontWeight: 600,
                color: '#1f2937',
                marginBottom: '1.5rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem',
                margin: '0 0 1.5rem 0'
              }}>
                <Mountain size={24} />
                Elevation Profile
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={prediction.elevation_profile}>
                  <defs>
                    <linearGradient id="elevationGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#667eea" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#667eea" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <XAxis 
                    dataKey="distance" 
                    type="number"
                    domain={['dataMin', 'dataMax']}
                    tickFormatter={(value) => value.toFixed(1)}
                    stroke="#9ca3af"
                    style={{ fontSize: '0.875rem' }}
                    label={{ value: 'Distance (km)', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis 
                    stroke="#9ca3af"
                    style={{ fontSize: '0.875rem' }}
                    label={{ value: 'Elevation (m)', angle: -90, position: 'insideLeft' }}
                  />
                  <Tooltip 
                    contentStyle={{
                      background: 'white',
                      border: '1px solid #e5e7eb',
                      borderRadius: '10px',
                      boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                    }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="elevation" 
                    stroke="#667eea" 
                    strokeWidth={3}
                    fill="url(#elevationGradient)" 
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Trail Map */}
            {prediction.trail_path && prediction.trail_path.length > 0 && (
              <div style={{
                background: 'white',
                borderRadius: '20px',
                padding: '2rem',
                boxShadow: '0 2px 8px rgba(0,0,0,0.06)',
                border: '1px solid #e5e7eb'
              }}>
                <h3 style={{
                  fontSize: '1.25rem',
                  fontWeight: 600,
                  color: '#1f2937',
                  marginBottom: '1.5rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  margin: '0 0 1.5rem 0'
                }}>
                  <MapPin size={24} />
                  Trail Map
                </h3>
                <div style={{ width: '100%', height: '400px', borderRadius: '12px', overflow: 'hidden' }}>
                  <iframe
                    srcDoc={`
                      <!DOCTYPE html>
                      <html>
                      <head>
                        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
                        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
                        <style>
                          body { margin: 0; padding: 0; }
                          #map { width: 100%; height: 400px; }
                        </style>
                      </head>
                      <body>
                        <div id="map"></div>
                        <script>
                          const coords = ${JSON.stringify(prediction.trail_path)};
                          const map = L.map('map').setView([coords[0][0], coords[0][1]], 13);
                          
                          L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                            attribution: '© OpenStreetMap contributors'
                          }).addTo(map);
                          
                          L.polyline(coords, {
                            color: '#dc2626',
                            weight: 4,
                            opacity: 0.8
                          }).addTo(map);
                          
                          const bounds = L.latLngBounds(coords);
                          map.fitBounds(bounds, { padding: [50, 50] });
                        </script>
                      </body>
                      </html>
                    `}
                    style={{ width: '100%', height: '400px', border: 'none' }}
                  />
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// Helper component for metric cards
function MetricCard({ icon: Icon, label, value, color }) {
  return (
    <div style={{
      background: '#f9fafb',
      padding: '1.25rem',
      borderRadius: '12px',
      borderLeft: `3px solid ${color}`
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
        <Icon size={20} color="#6b7280" />
        <span style={{ fontSize: '0.875rem', color: '#6b7280', fontWeight: 500 }}>{label}</span>
      </div>
      <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#1f2937' }}>
        {value}
      </div>
    </div>
  );
}