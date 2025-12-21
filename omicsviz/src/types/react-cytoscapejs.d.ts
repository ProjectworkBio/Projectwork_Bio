declare module 'react-cytoscapejs' {
  // Importing a generic type that represents any valid React component 
  // (functional or class-based), with a given props type.
  import type { ComponentType } from 'react';

  // Declared a constant called `CytoscapeComponent` that is of type `ComponentType<any>`.
  // Aka: "CytoscapeComponent is a React component that can accept *any* props."
  const CytoscapeComponent: ComponentType<any>;

  // Export component as the default export from the module.
  export default CytoscapeComponent;
}
