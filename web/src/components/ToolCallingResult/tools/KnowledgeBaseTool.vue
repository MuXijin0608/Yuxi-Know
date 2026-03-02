<template>
  <BaseToolCall :tool-call="toolCall" :hide-params="true">
    <template #header>
      <div class="sep-header">
        <span class="note">{{ operationLabel }}</span>
        <span class="separator" v-if="queryText">|</span>
        <span class="description">{{ queryText }}</span>
        <span class="separator" v-if="fileName">|</span>
        <span class="description" v-if="fileName">文件: {{ fileName }}</span>
      </div>
    </template>
    <template #result="{ resultContent }">
      <div v-if="operation === 'get_mindmap'" class="knowledge-base-result">
        <div class="mindmap-result">
          <pre class="mindmap-content">{{ formatMindmapResult(resultContent) }}</pre>
        </div>
      </div>

      <div v-else-if="isArrayResult(resultContent)" class="knowledge-base-result">
        <KbResultGroupedList :chunks="parsedData(resultContent)" />
      </div>

      <div v-else class="knowledge-base-result">
        <div class="plain-text-result">
          <pre class="plain-text-content">{{ resultContent }}</pre>
        </div>
      </div>
    </template>
  </BaseToolCall>
</template>

<script setup>
import { computed } from 'vue'
import BaseToolCall from '../BaseToolCall.vue'
import KbResultGroupedList from '@/components/sources/KbResultGroupedList.vue'

const props = defineProps({
  toolCall: {
    type: Object,
    required: true
  }
})

const args = computed(() => {
  const value = props.toolCall.args || props.toolCall.function?.arguments
  if (!value) return {}
  if (typeof value === 'object') return value
  try {
    return JSON.parse(value)
  } catch {
    return {}
  }
})

const toolName = computed(() => props.toolCall.name || props.toolCall.function?.name || '知识库')
const operation = computed(() => args.value.operation || 'search')

const operationLabel = computed(() => {
  const labels = {
    search: `${toolName.value} 搜索`,
    get_mindmap: toolName.value
  }
  return labels[operation.value] || operation.value
})

const queryText = computed(() => args.value.query_text || '')
const fileName = computed(() => args.value.file_name || '')

const parseData = (content) => {
  if (typeof content === 'string') {
    try {
      return JSON.parse(content)
    } catch {
      return []
    }
  }
  return content || []
}

const parsedData = (content) => parseData(content)

const isArrayResult = (content) => {
  if (Array.isArray(content)) return true
  if (typeof content === 'string') {
    try {
      return Array.isArray(JSON.parse(content))
    } catch {
      return false
    }
  }
  return false
}

const formatMindmapResult = (content) => {
  if (typeof content === 'string') return content
  if (typeof content === 'object') return JSON.stringify(content, null, 2)
  return String(content)
}
</script>

<style scoped lang="less">
.knowledge-base-result {
  background: var(--gray-0);
  border-radius: 8px;

  .mindmap-result,
  .plain-text-result {
    padding: 12px 16px;
    max-height: 300px;
    overflow-y: auto;
  }

  .mindmap-content,
  .plain-text-content {
    margin: 0;
    font-size: 13px;
    line-height: 1.6;
    color: var(--gray-700);
    white-space: pre-wrap;
    word-break: break-word;
  }

  .mindmap-content {
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  }
}
</style>
