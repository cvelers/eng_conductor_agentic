/**
 * Profile database loader — fetches and caches the European section database.
 */

let _profileDb = null;
let _loading = null;

/**
 * Load the profile database (lazy, cached).
 * @returns {Promise<Object>} The section database { IPE: {...}, HEB: {...}, HEA: {...} }
 */
export async function loadProfileDb() {
  if (_profileDb) return _profileDb;
  if (_loading) return _loading;

  _loading = fetch("/static/fea/profiles/european_sections.json")
    .then(r => {
      if (!r.ok) return fetch("/data/profiles/european_sections.json").then(r2 => r2.json());
      return r.json();
    })
    .then(data => {
      _profileDb = data;
      return data;
    })
    .catch(() => {
      // Fallback: try alternative path
      return fetch("../data/profiles/european_sections.json")
        .then(r => r.json())
        .then(data => { _profileDb = data; return data; })
        .catch(() => {
          console.warn("Could not load profile database");
          _profileDb = {};
          return {};
        });
    });

  return _loading;
}

/**
 * Get cached profile database (returns null if not yet loaded).
 */
export function getProfileDb() {
  return _profileDb;
}

/**
 * Look up a section by name (e.g., "IPE300", "HEB200").
 */
export function lookupSection(profileName) {
  if (!_profileDb) return null;
  const upper = profileName.toUpperCase();
  for (const series of Object.values(_profileDb)) {
    if (series[upper]) return { profileName: upper, ...series[upper] };
  }
  return null;
}
