import { NavLink } from "react-router-dom";
import { Home, Map, Route, Car, Brain, Settings } from "lucide-react";

const ORANGE = "#f97316";
const DARK_BG = "#0a0a0a";
const DARK_CARD = "#141414";
const DARK_BORDER = "#262626";
const DARK_TEXT_MUTED = "#737373";

const navItems = [
  { to: "/", icon: Home, label: "Home" },
  { to: "/vehicle", icon: Car, label: "Vehicle" },
  { to: "/map", icon: Map, label: "Map" },
  { to: "/analytics", icon: Brain, label: "Analytics" },
  { to: "/trips", icon: Route, label: "Trips" },
];

export default function Sidebar() {
  return (
    <aside
      className="w-20 flex flex-col items-center py-6"
      style={{
        backgroundColor: DARK_CARD,
        borderRight: `1px solid ${DARK_BORDER}`,
      }}
    >
      {/* Logo Placeholder */}
      <div className="mb-8">
        <div
          className="w-12 h-12 rounded-xl flex items-center justify-center"
          style={{ backgroundColor: DARK_BG }}
        >
          <img
            src="/images/porsche-logo.png"
            alt="Porsche"
            className="w-10 h-10 object-contain"
            onError={(e) => {
              e.currentTarget.style.display = "none";
              e.currentTarget.parentElement!.innerHTML = `<span style="font-weight:bold;color:${ORANGE};font-size:14px;">P</span>`;
            }}
          />
        </div>
        <div
          className="text-[8px] text-center mt-1 tracking-widest"
          style={{ color: DARK_TEXT_MUTED }}
        >
          PORSCHE
        </div>
      </div>

      {/* Navigation Icons */}
      <nav className="flex-1 flex flex-col gap-2">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-200 group relative
              ${isActive ? "text-white shadow-lg" : ""}`
            }
            style={({ isActive }) => ({
              backgroundColor: isActive ? ORANGE : "transparent",
              color: isActive ? "#fff" : DARK_TEXT_MUTED,
            })}
          >
            <Icon className="w-5 h-5" />
            {/* Tooltip */}
            <div
              className="absolute left-full ml-3 px-2 py-1 text-white text-xs rounded-md opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity whitespace-nowrap z-50"
              style={{
                backgroundColor: DARK_CARD,
                border: `1px solid ${DARK_BORDER}`,
              }}
            >
              {label}
            </div>
          </NavLink>
        ))}
      </nav>

      {/* Settings at bottom */}
      <NavLink
        to="/settings"
        className="w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-200 group relative"
        style={({ isActive }) => ({
          backgroundColor: isActive ? ORANGE : "transparent",
          color: isActive ? "#fff" : DARK_TEXT_MUTED,
        })}
      >
        <Settings className="w-5 h-5" />
        <div
          className="absolute left-full ml-3 px-2 py-1 text-white text-xs rounded-md opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity whitespace-nowrap z-50"
          style={{
            backgroundColor: DARK_CARD,
            border: `1px solid ${DARK_BORDER}`,
          }}
        >
          Settings
        </div>
      </NavLink>
    </aside>
  );
}
