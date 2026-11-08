import { describe, expect, it, vi } from 'vitest'
import { uploadInputMediaFile } from '@/lib/media-upload'
import { mergeOrUpdateTaskImages, uploadTaskImageFile } from '@/lib/task-image-utils'

describe('uploadInputMediaFile', () => {
  it('sends the requested input subfolder and overwrite=true by default to ComfyUI', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ name: 'voice.wav', subfolder: 'input/ref' }),
    })
    vi.stubGlobal('fetch', fetchMock)

    await uploadInputMediaFile(new File(['audio'], 'voice.wav', { type: 'audio/wav' }), 'input/ref')

    const body = fetchMock.mock.calls[0]?.[1]?.body as FormData
    expect(body.get('subfolder')).toBe('input/ref')
    expect(body.get('overwrite')).toBe('true')
  })

  it('uploads task images with overwrite=true and updates matching slots in-place without duplicating', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ name: 'office4.png', subfolder: '' }),
    })
    vi.stubGlobal('fetch', fetchMock)

    const uploaded = await uploadTaskImageFile(new File(['img2'], 'office4.png', { type: 'image/png' }))
    const body = fetchMock.mock.calls[0]?.[1]?.body as FormData
    expect(body.get('overwrite')).toBe('true')
    expect(uploaded.media_revision).toBeTruthy()

    const existing = [
      { id: 'slot-1', source_type: 'input' as const, file_path: 'office4.png', file_name: 'office4.png', shared_reference: true },
      { id: 'slot-2', source_type: 'input' as const, file_path: 'other.png', file_name: 'other.png' },
    ]
    const merged = mergeOrUpdateTaskImages(existing, [uploaded])
    expect(merged).toHaveLength(2)
    expect(merged[0].id).toBe('slot-1')
    expect(merged[0].shared_reference).toBe(true)
    expect(merged[0].media_revision).toBe(uploaded.media_revision)
  })
})
