import { h, type FunctionalComponent } from 'vue';
import { display } from '../lib/format';

// A token with visible whitespace, or a dimmed "empty" marker.
const TokText: FunctionalComponent<{ s: string }> = ({ s }) => {
  const d = display(s);
  return d ? d : h('span', { class: 'empty' }, 'empty');
};
TokText.props = ['s'];

export default TokText;
