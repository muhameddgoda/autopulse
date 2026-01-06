import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/layout/Layout';
import Home from './pages/Home';
import MapPage from './pages/MapPage';
import ChartsPage from './pages/ChartsPage';
import TripsPage from './pages/TripsPage';
import VehiclePage from './pages/VehiclePage';
import AnalyticsPage from './pages/AnalyticsPage';
import SafetyMonitor from './pages/SafetyMonitor';


function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="map" element={<MapPage />} />
          <Route path="charts" element={<ChartsPage />} />
          <Route path="analytics" element={<AnalyticsPage />} />
          <Route path="trips" element={<TripsPage />} />
          <Route path="vehicle" element={<VehiclePage />} />
          <Route path="/safety" element={<SafetyMonitor />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
