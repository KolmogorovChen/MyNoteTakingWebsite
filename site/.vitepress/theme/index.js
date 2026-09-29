import DefaultTheme from 'vitepress/theme';
import MermaidDiagram from './MermaidDiagram.vue';
import { useSectionNavigation } from './section-navigation';
import './style.css';
export default {
  extends: DefaultTheme,
  enhanceApp({ app }) { app.component('MermaidDiagram', MermaidDiagram); },
  setup() { useSectionNavigation(); }
};
