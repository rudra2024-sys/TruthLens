/**
 * Central "station" index for the app's core examination pages — gives each
 * one a consistent "position within the instrument" marker (e.g.
 * "Station 03 / 06") instead of every page inventing its own one-off
 * eyebrow text. Purely a display convention: not tied to routing, data, or
 * any backend concept of a "station".
 *
 * Home is Station 01 but isn't tagged on-page (it IS the entry point, and
 * LensNarrative's phase content is left untouched by this system). Login/
 * SignUp intentionally keep their own distinct "Access Terminal" framing
 * rather than being folded into the numbered stations — they're gates, not
 * examination stations.
 */
export const STATION_TOTAL = 6

export const stationTag = (index) =>
  `Station ${String(index).padStart(2, '0')} / ${String(STATION_TOTAL).padStart(2, '0')}`
