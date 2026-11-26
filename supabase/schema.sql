-- ============================================================================
-- EAE (Easy Arsenal Evaluation) — Supabase Database & Storage Schema (v5.33)
-- Idempotent migration for public.profiles, public.bullpen_sessions, and
-- trackman-pdfs Storage bucket with strict user-scoped RLS policies.
-- ============================================================================

-- 1. User Profiles Table
create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  role text,
  full_name text,
  organization text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.profiles enable row level security;

do $$
begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'public' and tablename = 'profiles' and policyname = 'Users can select own profile'
  ) then
    create policy "Users can select own profile"
      on public.profiles for select
      using (auth.uid() = id);
  end if;

  if not exists (
    select 1 from pg_policies
    where schemaname = 'public' and tablename = 'profiles' and policyname = 'Users can insert own profile'
  ) then
    create policy "Users can insert own profile"
      on public.profiles for insert
      with check (auth.uid() = id);
  end if;

  if not exists (
    select 1 from pg_policies
    where schemaname = 'public' and tablename = 'profiles' and policyname = 'Users can update own profile'
  ) then
    create policy "Users can update own profile"
      on public.profiles for update
      using (auth.uid() = id)
      with check (auth.uid() = id);
  end if;

  if not exists (
    select 1 from pg_policies
    where schemaname = 'public' and tablename = 'profiles' and policyname = 'Users can delete own profile'
  ) then
    create policy "Users can delete own profile"
      on public.profiles for delete
      using (auth.uid() = id);
  end if;
end $$;

-- 2. Bullpen Sessions Table (with athlete_folder)
create table if not exists public.bullpen_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  pitcher_name text not null,
  athlete_folder text default 'Unassigned',
  throws text not null default 'R',
  session_date text,
  source_type text default 'csv',
  total_pitches integer default 0,
  pitches_json jsonb not null,
  created_at timestamptz not null default now()
);

alter table public.bullpen_sessions
  add column if not exists athlete_folder text default 'Unassigned';

alter table public.bullpen_sessions enable row level security;

do $$
begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'public' and tablename = 'bullpen_sessions' and policyname = 'Users can select own bullpen_sessions'
  ) then
    create policy "Users can select own bullpen_sessions"
      on public.bullpen_sessions for select
      using (auth.uid() = user_id);
  end if;

  if not exists (
    select 1 from pg_policies
    where schemaname = 'public' and tablename = 'bullpen_sessions' and policyname = 'Users can insert own bullpen_sessions'
  ) then
    create policy "Users can insert own bullpen_sessions"
      on public.bullpen_sessions for insert
      with check (auth.uid() = user_id);
  end if;

  if not exists (
    select 1 from pg_policies
    where schemaname = 'public' and tablename = 'bullpen_sessions' and policyname = 'Users can update own bullpen_sessions'
  ) then
    create policy "Users can update own bullpen_sessions"
      on public.bullpen_sessions for update
      using (auth.uid() = user_id)
      with check (auth.uid() = user_id);
  end if;

  if not exists (
    select 1 from pg_policies
    where schemaname = 'public' and tablename = 'bullpen_sessions' and policyname = 'Users can delete own bullpen_sessions'
  ) then
    create policy "Users can delete own bullpen_sessions"
      on public.bullpen_sessions for delete
      using (auth.uid() = user_id);
  end if;
end $$;

-- 3. Raw TrackMan PDF Archive Bucket (INSERT-only for authenticated users)
insert into storage.buckets (id, name, public)
values ('trackman-pdfs', 'trackman-pdfs', false)
on conflict (id) do nothing;

do $$
begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'storage' and tablename = 'objects' and policyname = 'Authenticated users can upload trackman-pdfs'
  ) then
    create policy "Authenticated users can upload trackman-pdfs"
      on storage.objects for insert
      to authenticated
      with check (bucket_id = 'trackman-pdfs');
  end if;
end $$;
